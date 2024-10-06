use super::abstractor::Abstractor;
use super::centroid::Centroid;
use super::datasets::LargeSpace;
use super::datasets::SmallSpace;
use crate::cards::observation::Observation;
use crate::cards::street::Street;
use crate::clustering::abstraction::Abstraction;
use crate::clustering::histogram::Histogram;
use crate::clustering::metric::Metric;
use crate::clustering::progress::Progress;
use crate::clustering::xor::Pair;
use rand::distributions::Distribution;
use rand::distributions::WeightedIndex;
use rand::seq::IteratorRandom;
use rand::SeedableRng;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::RwLock;

/// Hierarchical K Means Learner
/// this is decomposed into the necessary data structures
/// for kmeans clustering to occur for a given `Street`.
/// it should also parallelize well, with kmeans and lookup
/// being the only mutable fields.
pub struct Hierarchical {
    street: Street,
    metric: Metric,
    points: LargeSpace,
    kmeans: Arc<RwLock<SmallSpace>>,
    lookup: Arc<RwLock<Abstractor>>,
}

impl Hierarchical {
    /// from scratch, generate and persist the full Abstraction lookup table
    pub fn upload() {
        log::info!("uploading abstraction lookup table");
        Self::outer()
            .inner() // turn
            .save()
            .inner() // flop
            .save();
    }

    /// start with the River layer. everything is empty because we
    /// can generate `Abstractor` and `SmallSpace` from "scratch".
    /// - `lookup`: lazy equity calculation of river observations
    /// - `kmeans`: equity percentile buckets of equivalent river observations
    /// - `metric`: absolute value of `Abstraction::Equity` difference
    /// - `points`: not used for inward projection. only used for clustering. and no clustering on River.
    fn outer() -> Self {
        Self {
            lookup: Arc::new(RwLock::new(Abstractor::default())),
            kmeans: Arc::new(RwLock::new(SmallSpace::default())),
            points: LargeSpace::default(),
            metric: Metric::default(),
            street: Street::Rive,
        }
    }
    /// hierarchically, recursively generate the inner layer
    fn inner(&self) -> Self {
        let inner = Self {
            lookup: Arc::new(RwLock::new(Abstractor::default())), // assigned during clustering
            kmeans: Arc::new(RwLock::new(SmallSpace::default())), // assigned during clustering
            street: self.inner_street(), // uniquely determined by outer layer
            metric: self.inner_metric(), // uniquely determined by outer layer
            points: self.inner_points(), // uniquely determined by outer layer
        };
        inner.initial();
        inner.cluster();
        inner
    }

    /// simply go to the previous street
    fn inner_street(&self) -> Street {
        log::info!("advancing from {} to {}", self.street, self.street.prev());
        self.street.prev()
    }
    /// compute the outer product of the `Abstraction -> Histogram`s at the current layer,
    /// - generate the _inner layer_ `Metric` between `Abstraction`s
    /// - by using the _outer layer_ `Metric` between `Histogram`s via EMD calcluations.
    ///
    /// we symmetrize the distance by averaging the EMDs in both directions.
    /// the distnace isn't symmetric in the first place only because our heuristic algo is not fully accurate
    fn inner_metric(&self) -> Metric {
        log::info!("computing metric {}", self.street);
        let lock = self.kmeans();
        let ref kmeans = lock.read().expect("poison").0;
        let mut metric = BTreeMap::new();
        for i in kmeans.keys() {
            for j in kmeans.keys() {
                if i > j {
                    let index = Pair::from((i, j));
                    let x = kmeans.get(i).expect("pre-computed").reveal();
                    let y = kmeans.get(j).expect("pre-computed").reveal();
                    let distance = self.metric.wasserstein(x, y) + self.metric.wasserstein(y, x);
                    let distance = distance / 2.0;
                    metric.insert(index, distance);
                }
            }
        }
        Metric(metric)
    }
    /// using the current layer's `Abstractor`,
    /// we generate the `LargeSpace` of `Observation` -> `Histogram`.
    /// 1. take all `Observation`s for `self.street.prev()`
    /// 2. map each to possible `self.street` `Observation`s
    /// 3. use `self.abstractor` to map each into an `Abstraction`
    /// 4. collect `Abstraction`s into a `Histogram`, for each `Observation`
    fn inner_points(&self) -> LargeSpace {
        log::info!("computing projections {}", self.street);
        use rayon::iter::IntoParallelIterator;
        use rayon::iter::ParallelIterator;
        let lock = self.lookup();
        let ref lookup = lock.read().expect("poison");
        LargeSpace(
            Observation::enumerate(self.street.prev())
                .into_par_iter()
                .map(|inner| (inner, lookup.projection(&inner)))
                .collect::<BTreeMap<Observation, Histogram>>(),
        )
    }

    /// initializes the centroids for k-means clustering using the k-means++ algorithm
    /// 1. choose 1st centroid randomly from the dataset
    /// 2. choose nth centroid with probability proportional to squared distance of nearest neighbors
    /// 3. collect histograms and label with arbitrary (random) `Abstraction`s
    ///
    /// if this becomes a bottleneck with contention,
    /// consider partitioning dataset or using lock-free data structures.
    fn initial(&self) {
        log::info!("initializing kmeans {}", self.street);
        let ref mut rng = rand::rngs::StdRng::seed_from_u64(self.street as u64);
        self.append(self.sample_uniform(rng));
        while self.k() > self.l() {
            log::info!("add initial {}", self.l());
            self.append(self.sample_outlier(rng));
        }
    }
    /// for however many iterations we want,
    /// 1. assign each `Observation` to the nearest `Centroid`
    /// 2. update each `Centroid` by averaging the `Observation`s assigned to it
    ///
    /// if this becomes a bottleneck with contention,
    /// consider partitioning dataset or using lock-free data structures.
    fn cluster(&self) {
        log::info!("clustering kmeans {}", self.street);
        for i in 0..self.t() {
            log::info!("assign and absorb {} {}", self.street, i);
            self.points
                .0
                .par_iter()
                .for_each(|(o, h)| self.update(o, h));
            log::info!("rotate centroids {} {}", self.street, i);
            self.kmeans()
                .write()
                .expect("poison")
                .0
                .par_iter_mut()
                .for_each(|(_, centroid)| centroid.rotate());
        }
    }

    /// mutation achieved by acquiring RwLock write access
    fn update(&self, observation: &Observation, histogram: &Histogram) {
        let ref abstraction = self.sample_neighbor(histogram);
        self.assign(abstraction, observation);
        self.absorb(abstraction, histogram);
    }
    /// absorb a `Histogram` into an `Abstraction`
    fn absorb(&self, abstraction: &Abstraction, histogram: &Histogram) {
        self.kmeans()
            .write()
            .expect("poisoned kmeans lock")
            .0
            .get_mut(abstraction)
            .expect("abstraction::from::neighbor::from::self.kmeans")
            .absorb(histogram);
    }
    /// assign an `Abstraction` to an `Observation`
    fn assign(&self, abstraction: &Abstraction, observation: &Observation) {
        self.lookup()
            .write()
            .expect("poisoned lookup lock")
            .0
            .insert(observation.clone(), abstraction.clone());
    }
    /// extending self.kmeans during intialization
    fn append(&self, histogram: Histogram) {
        self.kmeans()
            .write()
            .expect("poisoned kmeans lock")
            .0
            .insert(Abstraction::random(), Centroid::from(histogram));
    }

    /// the first Centroid is uniformly random across all `Observation` `Histogram`s
    fn sample_uniform(&self, rng: &mut rand::rngs::StdRng) -> Histogram {
        self.points
            .0
            .values()
            .choose(rng)
            .expect("observation projections have been populated")
            .clone()
    }
    /// each next Centroid is selected with probability proportional to
    /// the squared distance to the nearest neighboring Centroid.
    /// faster convergence, i guess. on the shoulders of giants
    fn sample_outlier(&self, rng: &mut rand::rngs::StdRng) -> Histogram {
        let weights = self
            .points
            .0
            .par_iter()
            .map(|(_, hist)| self.sample_distance(hist))
            .collect::<Vec<f32>>();
        let index = WeightedIndex::new(weights)
            .expect("valid weights array")
            .sample(rng);
        self.points
            .0
            .values()
            .nth(index)
            .expect("shared index with outer layer")
            .clone()
    }
    /// distance^2 to the nearest neighboring Centroid
    fn sample_distance(&self, histogram: &Histogram) -> f32 {
        self.kmeans()
            .read()
            .expect("poisoned kmeans lock")
            .0
            .par_iter()
            .map(|(_, centroid)| centroid.reveal())
            .map(|centroid| self.metric.wasserstein(histogram, centroid))
            .map(|min| min * min)
            .min_by(|dx, dy| dx.partial_cmp(dy).unwrap())
            .expect("find nearest neighbor")
    }
    /// find the nearest neighbor `Abstraction` to a given `Histogram`
    fn sample_neighbor(&self, histogram: &Histogram) -> Abstraction {
        self.kmeans()
            .read()
            .expect("poisoned kmeans lock")
            .0
            .par_iter()
            .map(|(abs, centroid)| (abs, centroid.reveal()))
            .map(|(abs, centroid)| (abs, self.metric.wasserstein(histogram, centroid)))
            .min_by(|(_, dx), (_, dy)| dx.partial_cmp(dy).unwrap())
            .expect("find nearest neighbor")
            .0
            .clone()
    }

    /// hyperparameter: how many centroids to learn
    fn k(&self) -> usize {
        match self.street {
            Street::Turn => 200,
            Street::Flop => 200,
            _ => unreachable!("how did you get here"),
        }
    }
    /// hyperparameter: how many iterations to run kmeans
    fn t(&self) -> usize {
        match self.street {
            _ => 100,
        }
    }
    /// length of current kmeans centroids
    fn l(&self) -> usize {
        self.kmeans() //
            .read()
            .expect("poisoned kmeans lock")
            .0
            .len()
    }

    /// write the full abstraction lookup table to disk
    /// 1. Write the PGCOPY header (15 bytes)
    /// 2. Write the flags (4 bytes)
    /// 3. Write the extension (4 bytes)
    /// 4. Write the observation and abstraction pairs
    /// 5. Write the trailer (2 bytes)
    fn save(self) -> Self {
        log::info!("uploading abstraction lookup table {}", self.street);
        use byteorder::BigEndian;
        use byteorder::WriteBytesExt;
        use std::fs::File;
        use std::io::Write;
        let mut file = File::create(format!("{}.pgcopy", self.street)).expect("new file");
        let lock = self.lookup();
        let ref lookup = lock.read().expect("poison").0;
        let mut progress = Progress::new(lookup.len(), 10);
        file.write_all(b"PGCOPY\n\xff\r\n\0").expect("header");
        file.write_u32::<BigEndian>(0).expect("flags");
        file.write_u32::<BigEndian>(0).expect("extension");
        for (observation, abstraction) in lookup.iter() {
            let obs = i64::from(*observation);
            let abs = i64::from(*abstraction);
            file.write_u16::<BigEndian>(2).expect("field count");
            file.write_u32::<BigEndian>(8).expect("8-bytes field");
            file.write_i64::<BigEndian>(obs).expect("observation");
            file.write_u32::<BigEndian>(8).expect("8-bytes field");
            file.write_i64::<BigEndian>(abs).expect("abstraction");
            progress.tick();
        }
        file.write_u16::<BigEndian>(0xFFFF).expect("trailer");
        self
    }
    /// read the full abstraction lookup table from disk
    /// 1. Skip PGCOPY header (15 bytes), flags (4 bytes), and header extension (4 bytes)
    /// 2. Read field count (should be 2)
    /// 3. Read observation length (4 bytes)
    /// 4. Read observation (8 bytes)
    /// 5. Read abstraction length (4 bytes)
    /// 6. Read abstraction (8 bytes)
    /// 7. Insert observation and abstraction into lookup table
    /// 8. Repeat until end of file
    pub fn load(street: Street) -> Abstractor {
        log::info!("downloading abstraction lookup table {}", street);
        use byteorder::BigEndian;
        use byteorder::ReadBytesExt;
        use std::fs::File;
        use std::io::BufReader;
        use std::io::Read;
        use std::io::Seek;
        use std::io::SeekFrom;
        let file = File::open(format!("{}.pgcopy", street)).expect("open file");
        let mut buffer = [0u8; 2];
        let mut lookup = BTreeMap::new();
        let mut reader = BufReader::new(file);
        reader.seek(SeekFrom::Start(23)).expect("seek past header");
        while reader.read_exact(&mut buffer).is_ok() {
            if u16::from_be_bytes(buffer) != 2 {
                break;
            }
            reader.read_u32::<BigEndian>().expect("observation length");
            let obs = reader.read_i64::<BigEndian>().expect("read observation");
            reader.read_u32::<BigEndian>().expect("abstraction length");
            let abs = reader.read_i64::<BigEndian>().expect("read abstraction");
            let observation = Observation::from(obs);
            let abstraction = Abstraction::from(abs);
            lookup.insert(observation, abstraction);
        }
        Abstractor(lookup)
    }

    /// thread-safe mutability for updating Abstraction table
    fn lookup(&self) -> Arc<RwLock<Abstractor>> {
        self.lookup.clone()
    }
    /// thread-safe mutability for kmeans centroids
    fn kmeans(&self) -> Arc<RwLock<SmallSpace>> {
        self.kmeans.clone()
    }
}
