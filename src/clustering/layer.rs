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

// Contains 1. the index of a specific point in the current Layer's 'points'
// field and 2. various additional information for said point needed to
// perform "Triangle inequality"-accelerated K-means clustering.
//j
// (see Elkan 2003 for more details)
#[derive(Debug, Clone)]
struct TriangleInequalityHelper {
    // The index in the current Layer kmeans for this point (i.e. a
    // specific value of 'x' in the paper)
    point_index: usize,
    // The index in the current Layer of the currently
    // assigned "nearest-neighbor" centroid for the specifed point ('c(x)' in
    // the paper) as well as the distance to said centroid
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

        // Effectively computing initial value for both c(x) and u(x) at the same time (closest initial center and
        // the upper bound on distance to closest initial center. See below.)
        let point_nearest_neighbors: Vec<Neighbor> =
            self.points().iter().map(|x| self.neighborhood(x)).collect();

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
                // "x"
                point_index: i,
                // "c(x)" (and distance to said c, since 'why not')
                nearest_neighbor: nearest_neighbor,
                // "l(x,c)"
                // "Set the lower bound l(x,c) = 0 for each point x and center c"
                lower_bounds: vec![0.0; self.street().k()],
                // "u(x)"
                // "Assign upper bounds x(x) = min_c d(x,c)" (which by
                //  definition is the distance of the nearest neighbor at
                //  this point)
                upper_bound: nearest_neighbor.1, })
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
        // The following 7-step algorithm is taken from Elkan (2003).
        // It uses triangle inequalities to accelerate the k-means
        // algorithm.
        // ****

        // *Step 1*: For all centers c and c', compute d(c,c'). For all centers
        // c, compute s(c) = (1/2) min_{c'!=c} d(c, c')
        //
        // This means s effectively contains the 'distance to the midpoint between
        // this centroid and the closest other centroid' for each centroid.

        // d(c, c')
        let centroid_to_centroid_distances = vec![vec![0.0; k]; k]; // might need to initialize to a negative number so we can tell when an entry isn't set?
                                                                    // ... orrrr if we don't NEED this later on, could stop keeping it indexed so neatly...
                                                                    // or even stop making it at all / just directly compute the s(c) vector... TBD.
                                                                    // Enumerate *first* before grabbing each (distinct) combination so that the
                                                                    // indices of each centroid we're looking at in the loop still map to their index
                                                                    // in Layer's kmeans field.
        for ((i1, c1), (i2, c2)) in self.kmeans().iter().enumerate().array_combinations() {
            let distance: f32 = 0.5 * self.emd(c1, c2);
            // By definiton they are the same distance from each other
            //
            // TODO: DOUBLE CHECK THAT THAT'S ACTUALLY THE CASE! (Assuming
            // it is, but I actually don't *know* that for certain).
            // (if not... then need to do 2 separate emd calculations)
            centroid_to_centroid_distances[i1][i2] = distance;
            centroid_to_centroid_distances[i2][i1] = distance;
        }
        // s(c) = (1/2) min_{c'!=c} d(c, c')
        let centroid_min_midpoint: Vec<f32> = centroid_to_centroid_distances
            .iter()
            .enumerate()
            // Figure out the mimum distance from each centroid to another centroid
            // (ie the closet other centroid)
            .map(|(i1, distances)| {
                distances
                    .iter()
                    .enumerate()
                    // Exclude the "0" distance from a centroid to itself before taking the min
                    .filter(|(i2, d)| i1 != i2)
                    .map(|(i2, d)| d)
                    .min()
                    .unwrap()
            })
            // Compute the distance to the midpoint instead of each other
            .map(|d| 0.5 * d)
            .collect();

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
