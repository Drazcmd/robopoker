use super::abstraction::Abstraction;
use super::histogram::Histogram;
use super::lookup::Lookup;
use super::metric::Metric;
use super::pair::Pair;
use super::transitions::Decomp;
use crate::cards::isomorphism::Isomorphism;
use crate::cards::isomorphisms::IsomorphismIterator;
use crate::cards::street::Street;
use crate::Energy;
use rand::distributions::Distribution;
use rand::distributions::WeightedIndex;
use std::collections::BTreeMap;

use std::collections::HashMap;

type Neighbor = (usize, f32);

pub struct Layer {
    street: Street,
    metric: Metric,
    kmeans: Vec<Histogram>, // positioned by K-means abstraction
    points: Vec<Histogram>, // positioned by Isomorphism
}

// "Carr[ied]... information" between k-means iterations for a specific point
// in self.points. See Elkan 2003 for more details.
//
// Used to accelerate k-means clustering via the paper's Triangle Inequality
// (abrv. 'TI' here) based optimized algorithm.
//
// NOTE: Includes some additional fields besides _just_ the bounds. (E.g. a
// field to help lookup the currently assigned centroid for the point).
#[derive(Debug, Clone)]
struct TIBounds {
    // The index into self.kmeans for the currently assigned centroid "nearest
    // neighbor" (i.e. c(x) in the paper) for this specifed point.
    assigned_centroid_idx: usize,
    // Lower bounds on the distance from this point to each centroid c
    // (l(x,c) in the paper).
    // Is k in length, where k is the number of centroids in the k-means
    // clustering. Each value inside the vector must correspond to the
    // same-indexed **centroid** (not point!) in the Layer.
    lower_bounds: Vec<f32>,
    // The upper bound on the distance from this point to its currently
    // assinged centroid (u(x) in the paper).
    upper_bound: f32,
    // Whether the upper_bound is out-of-date and needs a 'refresh'
    // (r(x) from the paper).
    stale_upper_bound: bool,
}

impl Layer {
    #[cfg(feature = "native")]
    /// all-in-one entry point for learning the kmeans abstraction and
    /// writing to disk in pgcopy
    pub fn learn() {
        use crate::save::upload::Table;
        Street::all()
            .into_iter()
            .rev()
            .filter(|&&s| !Self::done(s))
            .map(|&s| Self::grow(s).save())
            .count();
    }

    /// reference to the all points up to isomorphism
    fn points(&self) -> &Vec<Histogram> /* N */ {
        &self.points
    }
    /// reference to the current kmeans centorid histograms
    fn kmeans(&self) -> &Vec<Histogram> /* K */ {
        &self.kmeans
    }

    #[cfg(feature = "native")]
    /// primary clustering algorithm loop
    fn cluster(mut self) -> Self {
        log::info!("{:<32}{:<32}", "initialize  kmeans", self.street());
        let init = &mut self.init();
        let last = &mut self.kmeans;
        std::mem::swap(init, last);
        log::info!("{:<32}{:<32}", "clustering  kmeans", self.street());
        let t = self.street().t();
        log::info!("{:<32}{:<32}", " - # training iterations 't':", t);
        let progress = crate::progress(t);

        // Initialization from Elkan (2003) immediately prior to the 7-step
        // triangle inequality-based accelereated k-means algorithm.
        // """
        // First, pick initial centers. Set the lower bound CP% for each point
        // and center. Assign each to its closest initial center c(x) =
        // argmin_c d(x,c), using Lemma 1 to avoid redundant distance
        // calculations. Each time is computed, set l(x,c) = d(x,c). Assign
        // upper bounds u(x) = min_c d(x,c).
        // """
        let triangle_accelerate_todo_replaceme = true;
        let mut ti_helpers: Vec<TIBounds> = Vec::new();
        if triangle_accelerate_todo_replaceme {
            for helper in self
                .points()
                .iter()
                .map(|x| self.neighborhood(x))
                .map(|nearest_neighbor| TIBounds {
                    // "c(x)"'s index in self.kmeans()
                    assigned_centroid_idx: nearest_neighbor.0,
                    // "l(x,c)"
                    // "Set the lower bound l(x,c) = 0 for each point x and center c"
                    lower_bounds: vec![0.0; self.street().k()],
                    // "u(x)"
                    // "Assign upper bounds u(x) = min_c d(x,c)" (which by
                    //  definition is the distance of the nearest neighbor at
                    //  this point)
                    upper_bound: nearest_neighbor.1,
                    // "r(x)"
                    // (Not explicitly mentioned during the pre-step. But, we know that
                    // when starting out we literally _just_computed all the distances,
                    // so it should theoretically be safe to leave 'false' here.)
                    stale_upper_bound: false,
                })
                .collect::<Vec<_>>()
            {
                ti_helpers.push(helper);
            }
        }

        // WIP: Need to verify results are actually the same (and that it
        // really speeds things up in practice). As per the paper:
        //
        // """
        // We want the accelerated k-means algorithm to be usable wherever the
        // standard algorithm is used. Therefore, we need the accelerated
        // algorithm to satisfy three properties. First, it should be able to
        // start with any initial centers, so that all existing
        // initialization methods can continue to be used. Second, given the
        // same initial centers, it should al- ways produce exactly the same
        // final centers as the standard algorithm. Third, it should be able
        // to use any black-box distance metric, so it should not rely for
        // example on optimizations specific to Euclidean distance.
        //
        // Our algorithm in fact satisfies a condition stronger than the
        // second one above: after each iteration, it produces the same set
        // of center locations as the standard k-means method.
        // """
        for _ in 0..t {
            log::debug!("{:<32}{:<32}", "Starting training iteration:", t);

            if triangle_accelerate_todo_replaceme {
                let (ref next_kmeans, ref next_helpers) =
                    self.compute_next_centroids_ti_accl(&ti_helpers);

                let mut_kmeans = &mut self.kmeans();
                *mut_kmeans = next_kmeans;

                let mut_helpers = &mut (&ti_helpers);
                *mut_helpers = next_helpers
            } else {
                let ref next_kmeans = self.compute_next_centroids();

                let ref mut mut_kmeans = self.kmeans();
                *mut_kmeans = next_kmeans;
            }
            progress.inc(1);
        }
        progress.finish();
        println!();
        self
    }

    #[cfg(feature = "native")]
    /// initializes the centroids for k-means clustering using the k-means++ algorithm
    /// 1. choose 1st centroid randomly from the dataset
    /// 2. choose nth centroid with probability proportional to squared distance of nearest neighbors
    /// 3. collect histograms and label with arbitrary (random) `Abstraction`s
    fn init(&self) -> Vec<Histogram> /* K */ {
        use rand::rngs::SmallRng;
        use rand::SeedableRng;
        use rayon::iter::IntoParallelRefIterator;
        use rayon::iter::ParallelIterator;
        use std::hash::DefaultHasher;
        use std::hash::Hash;
        use std::hash::Hasher;
        // don't do any abstraction on preflop
        let k = self.street().k();
        let n = self.points().len();

        if self.street() == Street::Pref {
            assert!(n == k);
            return self.points().clone();
        }
        // deterministic pseudo-random clustering
        let ref mut hasher = DefaultHasher::default();
        self.street().hash(hasher);
        let ref mut rng = SmallRng::seed_from_u64(hasher.finish());
        // kmeans++ initialization
        let progress = crate::progress(k * n);
        let mut potentials = vec![1.; n];
        let mut histograms = Vec::new();
        while histograms.len() < k {
            let i = WeightedIndex::new(potentials.iter())
                .expect("valid weights array")
                .sample(rng);
            let x = self
                .points()
                .get(i)
                .expect("sharing index with outer layer");
            histograms.push(x.clone());
            potentials[i] = 0.;
            potentials = self
                .points()
                .par_iter()
                .map(|h| self.emd(x, h))
                .map(|p| p * p)
                .inspect(|_| progress.inc(1))
                .collect::<Vec<Energy>>()
                .iter()
                .zip(potentials.iter())
                .map(|(d0, d1)| Energy::min(*d0, *d1))
                .collect::<Vec<Energy>>();
        }
        progress.finish();
        println!();
        histograms
    }

    #[cfg(feature = "native")]
    /// calculates the next step of the kmeans iteration by
    /// determining K * N optimal transport calculations and
    /// taking the nearest neighbor
    fn compute_next_centroids(&self) -> Vec<Histogram> /* K */ {
        use rayon::iter::IntoParallelRefIterator;
        use rayon::iter::ParallelIterator;
        let k = self.street().k();
        let mut loss = 0f32;
        let mut centroids = vec![Histogram::default(); k];
        // assign points to nearest neighbors
        for (point, (neighbor, distance)) in self
            .points()
            .par_iter()
            .map(|h| (h, self.neighborhood(h)))
            .collect::<Vec<_>>()
            .into_iter()
        {
            loss = loss + distance * distance;
            centroids
                .get_mut(neighbor)
                .expect("index from neighbor calculation")
                .absorb(point);
        }
        log::debug!(
            "{:<32}{:<32}",
            "abstraction cluster RMS error",
            (loss / self.points().len() as f32).sqrt()
        );
        centroids
    }

    #[cfg(feature = "native")]
    /// WIP triangle-accelerated version of the 'next' function.
    /// Keep separate unless and until we've proven that this is
    /// going to actually be faster AND still correct
    ///
    /// calculates the next step of the kmeans iteration by
    /// determining up to K * N optimal transport calculations and
    /// taking the nearest neighbor, using triangle inequalities
    /// where possible to skip performing calculations
    ///
    /// TODO: This is currently entirely untested (aside from a
    /// couple of manual runs). Before replacing the original code
    /// we should first prove this works by running it against some
    /// of the datasets in the paper (Elkan (2003)) and verifying
    /// that we can replicate its results - as well as just
    /// generally writing some unit tests.
    fn compute_next_centroids_ti_accl(
        &self,
        ti_helpers: &[TIBounds],
    ) -> (
        Vec<Histogram>, /* K centroids */
        Vec<TIBounds>,  /* Updated Triangle Inequality Helpers */
    ) {
        // TODO: panic if the length of ti_helpers doesn't match the length of
        // self.points

        use rayon::iter::IndexedParallelIterator;
        use rayon::iter::IntoParallelRefIterator;
        use rayon::iter::ParallelIterator;

        let k = self.street().k();
        // TODO start tracking loss like in the other approach!
        // let mut loss = 0f32;

        // TODO: refactor / start using some things like this
        // let n = self.points().len();

        // ****
        // The following 7-step algorithm is taken from Elkan (2003). It uses
        // triangle inequalities to accelerate the k-means algorithm.
        // ****

        // *Step 1*: For all centers c and c', compute d(c,c'). For all
        //  centers c, compute s(c) = (1/2) min_{c'!=c} d(c, c')
        //
        // This means s effectively contains the 'distance to the midpoint
        // between this centroid and the closest other centroid' for each
        // centroid.

        log::info!("{:<32}", " - STEP 1 (remove me later)");

        // Step 1 (first half): d(c, c') for all centers c and c'
        let centroid_to_centroid_distances: Vec<Vec<f32>> = self
            .kmeans()
            .iter()
            // Get all combinations [(c1,c1), (c1,c2), ... (c_k, c_k)] into
            // a simple 1-D vector to allow for easily parallelizing the emd
            // calculations.
            // TLDR: effectively just itertools.array_combinations().
            .flat_map(|c| self.kmeans().iter().map(move |c_prime| (c, c_prime)))
            .collect::<Vec<_>>()
            .par_iter()
            .map(|(center1, center2)| self.emd(center1, center2)) // 1-D vector with length k^2
            .collect::<Vec<f32>>()
            .chunks(k) // Separate into k-length chunks so we can get it into a 2-D vector
            .map(|chunked| chunked.to_vec())
            .collect();

        // Step 1 (second half): s(c) = (1/2) min_{c'!=c} d(c, c')
        // (i.e. the closest midpoint to another centroid besides itself)
        let per_centroid_distance_to_closest_midpoint: Vec<f32> = centroid_to_centroid_distances
            .iter()
            .enumerate()
            .map(|(i, distances_from_centroid_i)| {
                // TLDR reducing down each per-centroid row to 1/2 the minimum
                // distance to all centroids except itself
                distances_from_centroid_i
                    .iter()
                    .enumerate()
                    .filter(|(other_centroid_index, _distance)| *other_centroid_index != i)
                    .map(|(_other_centroid_index, distance)| distance * 0.5)
                    // Workaround for f32 not implementing Ord due to NaN
                    // being incomparable.
                    // https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.min
                    .reduce(f32::min)
                    // ... TBD - might want to actually do something non-zero,
                    // seems like 0 could bite us if something "weird" were to
                    // happen here.
                    .unwrap_or(0.)
            })
            .collect();

        log::info!("{:<32}", " - STEP 2 (remove me later)");

        // Step 2: "Identify all points x such that u(x) <= s(c(x)).", i.e.
        // where the upper bound for the opint is less than its closest
        // midpoint.
        //
        // See also from the paper: "Logically, step (2) is redundant ...
        // [but c]omputationally step (2) is beneficial because if it
        // eliminates a point x from further consideration, then comparing u
        // (x) to l(x,c) for every c separately is not necessary."
        let step_2_excluded_points: Vec<usize> = ti_helpers
            .iter()
            .enumerate()
            .filter(|(_x, helper)| {
                // Note: s(c(x)), i.e. passing c(x) into s(c). So it's not the
                // index of the point itself that we should look up in s, but
                // rather the index of the _centroid to which the point x is
                // currently assigned_. Or in other words - the index of x's
                // current "nearest neighbor".
                //
                // TODO: THIS IS OBVIOUSLY CORRECT POST-INITIALIZATION, BUT
                // RELYING ON IT PAST THAT POINT MEANS WE NEED TO MAKE SURE
                // THAT THE HELPERS VECTOR CORRECTLY UPDATES THE NEAREST
                // NEIGHBOR FIELD TO POINT TO THE CURRENT CENTROID FOR EACH
                // POINT AFTER EACH ITERATION.(... probably need to stop
                // storing a Neighbor in the struct, since I think we'll end
                // up having to retake the distances again to do it cleanly,
                // defeating the purpose of this all. If that's correct
                // should instead just store the usize in the Helper struct)
                helper.upper_bound
                    <= per_centroid_distance_to_closest_midpoint[helper.assigned_centroid_idx]
            })
            .map(|(x, _)| x)
            .collect();
        log::info!("{:<32}", " - STEP 3 (remove me later)");

        // Step 3: For all remaining points x and centers c such that ...
        //
        // ** THIS VECTOR WILL BE UPDATED ON EACH ITERATION OF THE OUTER LOOP
        // OVER THE CENTROIDS BELOW **
        //
        // See paper as follows:
        // "In step (3), each time d(x, c) is calculated for any x and c, its
        //  lower bound is updated by assigning l(x, c) = d(x, c). Similarly,
        //  u(x) is updated whenever c(x) is changed or d(x, c(x)) is
        //  computed.
        //
        // Note also: "When step (3) is implemented with nested loops, the
        // outer loop can be over x or over c. For efficiency ... the outer
        // loop should be over c since k << n typically, and the inner loop
        // should be replaced by vectorized code that operates on all
        // relevant x collectively."
        //
        // This is a Hashmap instead of vector since some of the points are
        // excluded during this step (see step 2 above). Using a vector would
        // make things more complciated since there would be 'gaps' as a
        // result.
        // The Histograms are borrowed instead of owned since we don't need to
        // update them at all at this step; at most we just use them for some
        // ditance calculations. (The helpers however DO need to be owned
        // since we're replacing them all with the updated ones at the very
        // end of the step.)
        //
        // TODO: Possibly could just mutate helpers directly + immediately,
        // rather than creating a copy and doing the update at the end of
        // the step. (I _think_ this would allow us to avoid doing a bunch
        // of clones() throughout this step.) Depends on how Rayon works
        // though; might not actually be possible (TBD)
        //
        // Note: 1. This is the slowest part of the overall function (most of
        // our time is spent on this step), and 2. we're technically not
        // quite doing what the paper says. We should really be doing
        // vectorized code in the inner loop, but right now we're using rayon
        // parallelization. (This is currently the best we can do given that
        // we're using fancy `Histogram`s which need more complicated math to
        // compute the distance, as opposd to just doing standard euclidian
        // distance calculations.)
        log::info!("{:<32}", " - STEP 3, getting working points (remove me later)");
        let mut step_3_working_points: HashMap<usize, (&Histogram, TIBounds)> = self
            .points()
            .iter()
            .enumerate()
            .map(|(point_i, point_h)| (point_i, point_h, ti_helpers[point_i].clone()))
            .filter(|(point_i, _, _)| !step_2_excluded_points.contains(point_i))
            .map(|(point_i, point_h, helper)| (point_i, (point_h, helper)))
            .collect();

        // Note: looping over *all centers* here in the outer loop
        // (as mentioned above). NOT over the points / over anything in
        // step_3_working_points yet. (That all happens instead inside the
        // parallelized code down below inside this outer loop.)
        log::info!("{:<32}", " - STEP 3, starting outer loop (remove me later)");
        for (center_c_idx, center_c) in self.kmeans().iter().enumerate().collect::<Vec<_>>() {
            log::info!("{:<24}", "{:<8}", " - STEP 3, outer loop #", center_c_idx);
            let immutable_step_3_working_points = step_3_working_points.clone();
            for (point_i, point_h, helper) in immutable_step_3_working_points
                .par_iter()
                .map(|(point_i, histogram_and_helper)| {
                    (point_i, histogram_and_helper.0, &histogram_and_helper.1)
                })
                // ****
                // * STEP 3 FIRST HALF PER CENTROID: SETUP AND FILTERING (3.i, 3.ii, 3.iii) *
                // ****
                // Step 3 (i): ... [where] c != c(x)
                .filter(|(_, _, helper)| center_c_idx != helper.assigned_centroid_idx)
                // Step 3 (ii): ... [where] u(x) > l(x, c)
                .filter(|(_, _, helper)| helper.upper_bound > helper.lower_bounds[center_c_idx])
                // Step 3 (iii): ... [where] u(x) >  1/2 d(c(x), c)
                //
                // Note also from the paper:
                // "Condition (iii) inside step (3) is beneficial despite step (2), becaus
                // u(x) and c(x) may change during the execution of step (3)"
                .filter(|(_, _, helper)| {
                    // No need to recompute this distance since we already
                    // computed it back in step 1 (and given we haven't
                    // recomputed the centers yet those values are all
                    // still correct here).
                    // That said - in practice this doesn't meaningfully
                    // affect runtime either way (since we're doing all of
                    // them still in parallel, one per point). We could
                    // alternatively just compute these from scratch without
                    // it slowing things down at all.
                    let dist_between_centroids =
                        &centroid_to_centroid_distances[helper.assigned_centroid_idx][center_c_idx];
                    helper.upper_bound > 0.5 * dist_between_centroids
                })
                // ****
                // * STEP 3 SECOND HALF PER CENTROID: DISTANCE COMPUTATIONS AND UPDATES (3.a and 3.b) *
                // ****
                // Step 3.a: If r(x) then compute d(x, c(x)) and assign r(x) =
                // false. Otherwise, d(x, c(x)) = u(x).
                .map(|(point_i, point_h, helper)| {
                    let possibly_updated_helper_and_distance_from_point_to_current_centroid: (
                        TIBounds,
                        f32,
                    ) = if helper.stale_upper_bound {
                        let mut h: TIBounds = helper.clone();
                        let distance_point_to_current_centroid: f32 =
                            self.emd(point_h, &self.kmeans()[helper.assigned_centroid_idx]);
                        h.upper_bound = distance_point_to_current_centroid;
                        // As discussed above: "each time d(x, c) is
                        // calculated for any x and c, its lower bound is
                        // updated by assigning l(x, c) = d(x, c)" and
                        // "u(x) is updated whenever c(x) is changed or d
                        //  (x, c(x)) is computed."
                        h.lower_bounds[center_c_idx] = distance_point_to_current_centroid;
                        // Step 3.a: If r(x) then compute d(x, c(x)) and assign r(x) =
                        // false. Otherwise, d(x, c(x)) = u(x).
                        h.stale_upper_bound = false;

                        (h, distance_point_to_current_centroid)
                    } else {
                        (helper.clone(), helper.upper_bound)
                    };
                    (
                        point_i,
                        point_h,
                        possibly_updated_helper_and_distance_from_point_to_current_centroid.0,
                        possibly_updated_helper_and_distance_from_point_to_current_centroid.1,
                    )
                })
                // Step 3.b:
                // If d(x, c(x)) > l(x,c)
                // or d(x, c(x)) > (1/2) d(c(x), c)
                // then:
                //  Compute d(x,c)
                //  If d(x,c) < d(x, c(x)) then assign c(x) = c
                .map(
                    |(point_i, point_h, helper, distance_point_to_current_centroid)| {
                        let mut out_helper = helper.clone();
                        // If d(x, c(x)) > l(x,c)
                        // or d(x, c(x)) > (1/2) d(c(x), c)
                        if distance_point_to_current_centroid > helper.lower_bounds[center_c_idx]
                            || distance_point_to_current_centroid >  // (1/2) * d(c(x), c)
                        0.5 * centroid_to_centroid_distances[
                            helper.assigned_centroid_idx][center_c_idx]
                        {
                            // ... Compute d(x,c)
                            let distance_point_to_center_c = self.emd(point_h, center_c);
                            // As discussed above: "each time d(x, c) is
                            // calculated for any x and c, its lower bound is
                            // updated by assigning l(x, c) = d(x, c)" and
                            // "u(x) is updated whenever c(x) is changed or d
                            //  (x, c(x)) is computed."
                            out_helper.lower_bounds[center_c_idx] = distance_point_to_center_c;

                            // ... If d(x,c) < d(x, c(x)) then assign c(x) = c
                            if distance_point_to_center_c < distance_point_to_current_centroid {
                                out_helper.assigned_centroid_idx = center_c_idx;
                            }
                        }
                        (point_i, point_h, out_helper)
                    },
                )
                .collect::<Vec<_>>()
                .into_iter()
            {
                step_3_working_points.insert(*point_i, (point_h, helper));
            }
        }

        log::info!("{:<32}", " - STEP 4 (remove me later)");

        // Merge the updated helper values back with the original vector we got
        // at the start of the function (which has entries for *all* points, not
        // just the ones bieng updated in step 3).
        let step_4_helpers: Vec<&TIBounds> = ti_helpers
            .iter()
            .enumerate()
            .map(|(point_i, original_helper)| {
                if step_3_working_points.contains_key(&point_i) {
                    &(step_3_working_points[&point_i].1)
                } else {
                    original_helper
                }
            })
            .collect();

        // Step 4: For each center c, let m(c) be the mean of the points
        // assigned to c.
        //
        // This becomes the new centroid for the next step:
        // """
        // Step 4 computes the new lcoation fo each cluster center c.
        // Setting m(c) to be the mean of the points assigned to c is
        // appropriate when the distance metric in use is Euclidean distance.
        // Otherwise m(c) may be defined differently...
        // """
        //
        // Note also:
        // """
        // Step 4 computes the new location of each cluster center.
        // Setting m(c) to be the mean of the points assigned to is
        // appropriate when the distance metric in use is Euclidean
        // distance. Otherwise, may be defined differently. For
        // example, with k-medians the new center of each cluster is
        // a representative member of the cluster.
        // """
        //
        // In this case it's a little weird looking ('aborbing' histograms) since we're using emd
        // instead of Euclidean distance.
        let points_assigned_per_center: Vec<Vec<&Histogram>> = self
            .kmeans()
            .iter()
            .enumerate()
            .map(|(center_c_idx, _center_c)| {
                step_4_helpers
                    .iter()
                    .enumerate()
                    .filter(|(_point_i, helper)| helper.assigned_centroid_idx == center_c_idx)
                    .map(|(point_i, _)| &self.points()[point_i])
                    .collect()
            })
            .collect();
        // let mut loss = 0f32;
        let mut new_centroids: Vec<Histogram> = vec![];
        for points in points_assigned_per_center.iter() {
            let mut mean_of_assigned_points = points[0].clone();
            if points.is_empty() {
                // TODO: Figure out what to do for the centroid if there's no poitns assigned to it.
                log::error!("No points assigned to current centroid. This is currently an edge case we are unable to resolve; for more details see https://github.com/krukah/robopoker/issues/34#issuecomment-2860641178")
            }
            for point in points.iter().skip(1) {
                mean_of_assigned_points.absorb(point);
            }
            let next_centroid = mean_of_assigned_points;
            new_centroids.push(next_centroid);
        }

        log::info!("{:<32}", " - STEP 5 (remove me later)");

        // Step 5: Update lower bounds. From paper: ""
        // 5. For each point x and center c, assign
        //    l(x,c) = max{ l(x, c) - d(c, m(c)), 0 }
        // """
        //
        // Optimization: compute distance from each centroid to the its
        // replacement *one time* rather than inside the per-point loops of
        // steps 5 and 6.
        // (Note: this is highly differant than step 1! This isn't computing
        // each centroid to *all* the replacements, only *its* specific
        // replacement (i.e. the one with the same index). Hence it's a
        // Vec<f32> of length k, NOT a Vec<Vec<f32>>.
        //
        // TODO: double check that distance is equivalent in either direction.
        // (If not, would need to e.g. compute a Vec<(f32, f32)> instead.)
        let new_centroid_movements: Vec<f32> = new_centroids
            .par_iter()
            .zip(&self.kmeans)
            .map(|(old_center, new_center)| self.emd(old_center, new_center))
            .collect();

        let mut step_5_helpers: Vec<TIBounds> = step_4_helpers.into_iter().cloned().collect();
        for helper in &mut step_5_helpers {
            helper.lower_bounds = helper
                .lower_bounds
                // TODO: investigate whether we should just do this single threaded
                // given we're no longer doing any emd calculations here.
                // (And/or consider doing a vectorized update.)
                .par_iter()
                .enumerate()
                .map(|(center_c_idx, lower_bound)| {
                    // d(c, m(c))
                    // (Must do a distance calc since m(c) is new / not in the
                    // original set of centroids we computed distances for at
                    // the start)
                    let dist_center_and_new_center = &new_centroid_movements[center_c_idx];
                    f32::max(
                        // l(x,c) - d(c, m(c))
                        lower_bound - dist_center_and_new_center,
                        0.0,
                    )
                })
                .collect();
        }

        log::info!("{:<32}", " - STEP 6 (remove me later)");
        // Step 6: Update upper bounds. From paper: """
        // 6. For each point x, assign
        //    u(x) = u(x) + d(m(c(x)), c(x))
        //    r(x) = true
        // """
        // TODO refactor probably can get away with continuing to borrow here.
        // And/or do using a .map() inside in a .par_iter() etc.
        let mut step_6_helpers: Vec<TIBounds> = step_5_helpers;
        for helper in &mut step_6_helpers {
            // u(x) = u(x) + d(m(c(x)), c(x))
            // TODO: VERIFY THAT d(m(c(x)), c(x)) = d(c(x), m(c(x))).
            // IF NOT THEN THIS WILL BE INCORRECT.
            let dist_center_and_new_center = &new_centroid_movements[helper.assigned_centroid_idx];
            helper.upper_bound += dist_center_and_new_center;
            // r(x) = true
            helper.stale_upper_bound = true;
        }

        // Form paper "[Compute] the new location of each cluster center",
        // i.e. Step 7:
        // "7. Replace each center c by m(c)"
        log::info!("{:<32}", " - STEP 7 (remove me later)");
        (new_centroids, step_6_helpers)
    }

    /// wrawpper for distance metric calculations
    fn emd(&self, x: &Histogram, y: &Histogram) -> Energy {
        self.metric.emd(x, y)
    }
    /// because we have fixed-order Abstractions that are determined by
    /// street and K-index, we should encapsulate the self.street depenency
    fn abstraction(&self, i: usize) -> Abstraction {
        Abstraction::from((self.street(), i))
    }
    /// calculates nearest neighbor and separation distance for a Histogram
    fn neighborhood(&self, x: &Histogram) -> Neighbor {
        self.kmeans()
            .iter()
            .enumerate()
            .map(|(k, h)| (k, self.emd(x, h)))
            .min_by(|(_, dx), (_, dy)| dx.partial_cmp(dy).unwrap())
            .expect("find nearest neighbor")
            .into()
    }

    /// reference to current street
    fn street(&self) -> Street {
        self.street
    }
    /// take outer triangular product of current learned kmeans
    /// Histograms, using whatever is stored as the future metric
    fn metric(&self) -> Metric {
        log::info!("{:<32}{:<32}", "calculating metric", self.street());
        let mut metric = BTreeMap::new();
        for (i, x) in self.kmeans.iter().enumerate() {
            for (j, y) in self.kmeans.iter().enumerate() {
                if i > j {
                    let ref a = self.abstraction(i);
                    let ref b = self.abstraction(j);
                    let index = Pair::from((a, b));
                    let distance = self.metric.emd(x, y) + self.metric.emd(y, x);
                    let distance = distance / 2.;
                    metric.insert(index, distance);
                }
            }
        }
        Metric::from(metric)
    }
    /// in ObsIterator order, get a mapping of
    /// Isomorphism -> Abstraction
    #[cfg(feature = "native")]
    fn lookup(&self) -> Lookup {
        log::info!("{:<32}{:<32}", "calculating lookup", self.street());
        use crate::save::upload::Table;
        use rayon::iter::IntoParallelRefIterator;
        use rayon::iter::ParallelIterator;
        let street = self.street();
        match street {
            Street::Pref | Street::Rive => Lookup::grow(street),
            Street::Flop | Street::Turn => self
                .points()
                .par_iter()
                .map(|h| self.neighborhood(h))
                .collect::<Vec<Neighbor>>()
                .into_iter()
                .map(|(k, _)| self.abstraction(k))
                .zip(IsomorphismIterator::from(street))
                .map(|(abs, iso)| (iso, abs))
                .collect::<BTreeMap<Isomorphism, Abstraction>>()
                .into(),
        }
    }
    /// in AbsIterator order, get a mapping of
    /// Abstraction -> Histogram
    /// end-of-recurse call
    fn decomp(&self) -> Decomp {
        log::info!("{:<32}{:<32}", "calculating transitions", self.street());
        self.kmeans()
            .iter()
            .cloned()
            .enumerate()
            .map(|(k, centroid)| (self.abstraction(k), centroid))
            .collect::<BTreeMap<Abstraction, Histogram>>()
            .into()
    }
}

#[cfg(feature = "native")]
impl crate::save::upload::Table for Layer {
    fn done(street: Street) -> bool {
        Lookup::done(street) && Decomp::done(street) && Metric::done(street)
    }
    fn save(&self) {
        self.metric().save();
        self.lookup().save();
        self.decomp().save();
    }
    fn grow(street: Street) -> Self {
        let layer = match street {
            Street::Rive => Self {
                street,
                kmeans: Vec::default(),
                points: Vec::default(),
                metric: Metric::default(),
            },
            _ => Self {
                street,
                kmeans: Vec::default(),
                points: Lookup::load(street.next()).projections(),
                metric: Metric::load(street.next()),
            },
        };
        layer.cluster()
    }

    fn name() -> String {
        unimplemented!()
    }
    fn copy() -> String {
        unimplemented!()
    }
    fn load(_: Street) -> Self {
        unimplemented!()
    }
    fn creates() -> String {
        unimplemented!()
    }
    fn indices() -> String {
        unimplemented!()
    }
    fn columns() -> &'static [tokio_postgres::types::Type] {
        unimplemented!()
    }
    fn sources() -> Vec<String> {
        unimplemented!()
    }
}
