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

type Neighbor = (usize, f32);

pub struct Layer {
    street: Street,
    metric: Metric,
    kmeans: Vec<Histogram>, // positioned by K-means abstraction
    points: Vec<Histogram>, // positioned by Isomorphism
}

// Contains for a specific point in the current Layer's 'points' field various
// additional information for said point needed to perform "Triangle
// inequality"-accelerated K-means clustering.
//
// Intended use case is storing in a vector where each value ina the vector
// corresponds to the Point with matching index in the current Layer's kmeans
// field.
//
// (see Elkan 2003 for more details)
#[derive(Debug, Clone)]
struct TriangleInequalityHelper {
    // 0. The index into self.kmeans of c(x) in the paper - i.e. the index in the
    // current Layer of the currently assigned "nearest-neighbor" centroid
    // for this specifed point.
    //
    // 1. The distance from x to said centroid c(x).
    //
    // TODO: DECIDE WHETHER TO STORE THE ACTUAL HISTOGRAM HERE(... or
    // to *just* store the index and discard the f32 distance. TBD how much
    // work it is to keep track of them...)
    nearest_neighbor: Neighbor,
    // Lower bounds on the distance from this point to each centroid c
    // (l(x,c) in the paper).
    // Is k in length, where k is the number of centroids in the k-means
    // clustering. Each value inside the vector must correspond to the
    // same-indexed centroid in the Layer.
    lower_bounds: Vec<f32>,
    // The upper bound on the distance from this point to its currently
    // assinged centroid (u(x) in the paper).
    upper_bound: f32,
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
        let ref mut init = self.init();
        let ref mut last = self.kmeans;
        std::mem::swap(init, last);
        log::info!("{:<32}{:<32}", "clustering  kmeans", self.street());
        let t = self.street().t();
        let progress = crate::progress(t);
        let triangle_accelerate_todo_replaceme = false;

        // Initialization from Elkan (2003) immediately prior to the 7-step
        // triangle inequality-based accelereated k-means algorithm.
        // """
        // First, pick initial centers. Set the lower bound CP% for each point
        // and center. Assign each to its closest initial center c(x) =
        // argmin_c d(x,c), using Lemma 1 to avoid redundant distance
        // calculations. Each time is computed, set l(x,c) = d(x,c). Assign
        // upper bounds u(x) = min_c d(x,c).
        // """
        let triangle_inequality_helpers: Vec<TriangleInequalityHelper> = self
            .points()
            .iter()
            .map(|x| self.neighborhood(x))
            .enumerate()
            .map(|(i, nearest_neighbor)| TriangleInequalityHelper {
                // "c(x)" (and distance to said c, since 'why not')
                nearest_neighbor: nearest_neighbor,
                // "l(x,c)"
                // "Set the lower bound l(x,c) = 0 for each point x and center c"
                lower_bounds: vec![0.0; self.street().k()],
                // "u(x)"
                // "Assign upper bounds x(x) = min_c d(x,c)" (which by
                //  definition is the distance of the nearest neighbor at
                //  this point)
                upper_bound: nearest_neighbor.1,
            })
            .collect();

        for _ in 0..t {
            if triangle_accelerate_todo_replaceme {
                // TODO I assume the extra clone() calls are probably not the right way to do this / are wasteful.
                // ... or maybe not, maybe that's ok. Not sure actually! Need to go learn more about rust to know for certain
                // what I'm 'meant' to be doing in cases like these....

                // NEED TO UPDATE THIS I THINK. Need to have a way where AT THE START
                // we initialize "c(x)" mapping each point to its "closest initial center"
                // so can't do it inside the function. Meaning should proabbly be yet another
                // input we pass in like lower and upper vectors...
                // ... OH. Also I think scoping means this potentially never actually 'does'
                // anything for the next loop (not sure... tbd. Need to go read up on rust)
                let (ref mut next, triangle_inequality_helpers) =
                    self.next_kmeans_iteration2_accl(triangle_inequality_helpers.clone());
                let ref mut last = self.kmeans;
                std::mem::swap(next, last);
            } else {
                let ref mut next = self.next_kmeans_iteration();
                let ref mut last = self.kmeans;
                std::mem::swap(next, last);
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
    fn next_kmeans_iteration(&self) -> Vec<Histogram> /* K */ {
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
    fn next_kmeans_iteration2_accl(
        &self,
        triangle_inequality_helpers: Vec<TriangleInequalityHelper>,
    ) -> (
        Vec<Histogram>,                /* K centroids */
        Vec<TriangleInequalityHelper>, /* Updated Triangle Inequality Helpers */
    ) {
        use rayon::iter::IntoParallelRefIterator;
        use rayon::iter::ParallelIterator;
        let k = self.street().k();
        let mut loss = 0f32;
        let mut output_centroids = vec![Histogram::default(); k];

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

        // 1.1: d(c, c') for all centers c and c'
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

        // 1.2: s(c) = (1/2) min_{c'!=c} d(c, c')
        // (i.e. the closest midpoint to another centroid besides itself)
        let per_centroid_distance_to_closet_midpoint: Vec<f32> = centroid_to_centroid_distances
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

        // Step 2: "Identify all points x such that u(x) <= s(c(x)).", i.e.
        // where the upper bound for the opint is less than its closest
        // midpoint.
        //
        // See also from the paper: "Logically, step (2) is redundant ...
        // [but c]omputationally step (2) is beneficial because if it
        // eliminates a point x from further consideration, then comparing u
        // (x) to l(x,c) for every c separately is not necessary."
        let step_2_excluded_points: Vec<usize> = triangle_inequality_helpers
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
                    <= per_centroid_distance_to_closet_midpoint[helper.nearest_neighbor.0]
            })
            .map(|(x, _)| x)
            .collect();

        // Step 3: For all remaining points x and centers c such that ...
        //
        // ** This vector will be mutated during step 3 !! **
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
        let step_3_working_points: Vec<(usize, &Histogram, TriangleInequalityHelper)> = self
            .points()
            .iter()
            .enumerate()
            // TBD: Do we really really want to do a clone here? Need to dig into
            // rust a bit more...
            .map(|(i, h)| (i, h, triangle_inequality_helpers[i].clone()))
            .filter(|(x, _, _)| !step_2_excluded_points.contains(x))
            .collect();
        // TODO: create r(x) for tracking when u(x) is out of date
        // let step_3_upper_bound_stale = vec![.......]

        for center_c in self.kmeans() {
            let per_center_c_step_3_points: Vec<&(usize, &Histogram, TriangleInequalityHelper)> =
                step_3_working_points
                    .iter()
                    // Step 3 (i): ... [where] c != c(x)
                    .filter(|(i, _h, helpers)| *i != helpers.nearest_neighbor.0)
                    // Step 3 (ii): ... [where] u(x) > l(x, c)
                    .filter(|(i, _h, helpers)| helpers.upper_bound > helpers.lower_bounds[*i])
                    // Step 3 (iii): ... [where] u(x) >  1/2 d(c(x), c)
                    //
                    // Note also from the paper:
                    // "Condition (iii) inside step (3) is beneficial despite step (2), becaus
                    // u(x) and c(x) may change during the execution of step (3)"
                    .filter(|(_i, _h, helpers)| {
                        let distance_to_midpoint_of_current_centroid_and_center_c =
                            0.5 * self.emd(&self.kmeans[helpers.nearest_neighbor.0], center_c);
                        return helpers.upper_bound
                            > distance_to_midpoint_of_current_centroid_and_center_c;
                    })
                    .collect();

            // (As discussed above: each time we compute d(x,c) we update
            // the value l(x,c).)

            // Step 3.a: If r(x) then compute d(x, c(x)) and assign r(x) =
            // false. Otherwise, d(x, c(x)) = u(x).
        }

        // Step 4: For each center c, let m(c) be the mean of the points
        // assigned to c

        // Update lower bounds. From paper: ""
        // 5. For each point x and center c, assign
        //    l(x,c) = max{ l(x, c) - d(c, m(c)), 0 }
        // """
        //

        // Update upper bounds. From paper: """
        //    u(x) = u(x) + d(m(c(x)), c(x))
        //    r(x) = true
        // """
        // Notably, no need to mess with r(x) since we're
        // already inside a loop that sets it
        //
        // ... this is kinda weird actually, since in the middle
        // of the iteration we'll be messing with l and u...
        // Both "each time d(x,c) is computed set l(x,c) = d(x,c)"
        // and "u(x) ... may change during the executino of step(3)"
        // (which is itself weird, I don't see any references to updating
        // it there or anywhere else so far...?)
        //
        // arguably it's kinda weird that we're passing it in
        // and out of the function like this. Though, in practice
        // I think is kinda nice to constrain the mutations a bit
        // ... so maybe this is ok after all.
        (output_centroids, triangle_inequality_helpers)
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
