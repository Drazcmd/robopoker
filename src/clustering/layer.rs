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
use crate::Save;
use rand::distributions::Distribution;
use rand::distributions::WeightedIndex;
use std::collections::BTreeMap;

type Neighbor = (usize, f32);

pub struct Layer {
    street: Street,
    metric: Metric,
    points: Vec<Histogram>, // positioned by Isomorphism
    kmeans: Vec<Histogram>, // positioned by K-means abstraction
}

impl Layer {
    /// primary clustering algorithm loop
    fn cluster(mut self) -> Self {
        log::info!("{:<32}{:<32}", "initialize kmeans", self.street());
        let ref mut init = self.init();
        let ref mut last = self.kmeans;
        std::mem::swap(init, last);
        log::info!("{:<32}{:<32}", "clustering kmeans", self.street());
        let t = self.street().t();
        let progress = crate::progress(t);
        for _ in 0..self.street().t() {
            let ref mut next = self.next();
            let ref mut last = self.kmeans;
            std::mem::swap(next, last);
            progress.inc(1);
        }
        progress.finish();
        self
    }

    /// reference to the observed points
    fn points(&self) -> &Vec<Histogram> /* N */ {
        &self.points
    }
    /// reference to the current kmeans centorid histograms
    fn kmeans(&self) -> &Vec<Histogram> /* K */ {
        &self.kmeans
    }

    /// initializes the centroids for k-means clustering using the k-means++ algorithm
    /// 1. choose 1st centroid randomly from the dataset
    /// 2. choose nth centroid with probability proportional to squared distance of nearest neighbors
    /// 3. collect histograms and label with arbitrary (random) `Abstraction`s
    fn init(&self) -> Vec<Histogram> /* K */ {
        use rayon::iter::IntoParallelRefIterator;
        use rayon::iter::ParallelIterator;
        let n = self.points().len();
        let k = self.street().k();
        let progress = crate::progress(k * n);
        let mut histograms = Vec::new();
        let mut potentials = vec![1.; n];
        let ref mut rng = rand::thread_rng();
        while histograms.len() < self.street().k() {
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
        histograms
    }
    /// calculates the next step of the kmeans iteration by
    /// determining K * N optimal transport calculations and
    /// taking the nearest neighbor
    fn next(&self) -> Vec<Histogram> /* K */ {
        use rayon::iter::IntoParallelRefIterator;
        use rayon::iter::ParallelIterator;
        //? check for empty centroids??
        let next = vec![Histogram::default(); self.street().k()];
        self.points()
            .par_iter()
            .map(|h| (h, self.neighboring(h)))
            .collect::<Vec<(&Histogram, Neighbor)>>()
            .into_iter()
            .fold(next, |mut kmeans, (hist, (mean, _))| {
                kmeans[mean].absorb(hist);
                kmeans
            })
    }

    /// wrawpper for distance metric calculations
    fn emd(&self, x: &Histogram, y: &Histogram) -> Energy {
        self.metric.emd(x, y)
    }
    /// because we have fixed-order Abstractions that are determined by
    /// street and K-index, we should encapsulate the self.street depenency
    fn abstracting(&self, i: usize) -> Abstraction {
        Abstraction::from((self.street(), i))
    }
    /// calculates nearest neighbor and separation distance for a Histogram
    fn neighboring(&self, x: &Histogram) -> Neighbor {
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
    /// take outer product of current learned kmeans
    /// Histograms, using whatever is stored as the future metric
    fn metric(&self) -> Metric {
        log::info!("{:<32}{:<32}", "calculating metric", self.street());
        let mut metric = BTreeMap::new();
        for (i, x) in self.kmeans.iter().enumerate() {
            for (j, y) in self.kmeans.iter().enumerate() {
                if i > j {
                    let ref a = self.abstracting(i);
                    let ref b = self.abstracting(j);
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
    fn lookup(&self) -> Lookup {
        log::info!("{:<32}{:<32}", "calculating lookup", self.street());
        use rayon::iter::IntoParallelRefIterator;
        use rayon::iter::ParallelIterator;
        let n = self.street().n_isomorphisms();
        let street = self.street();
        let progress = crate::progress(n);
        match street {
            Street::Pref | Street::Rive => Lookup::make(street),
            Street::Flop | Street::Turn => self
                .points()
                .par_iter()
                .map(|h| self.neighboring(h))
                .inspect(|_| progress.inc(1))
                .collect::<Vec<Neighbor>>()
                .into_iter()
                .map(|(k, _)| self.abstracting(k))
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
            .map(|(k, mean)| (self.abstracting(k), mean))
            .collect::<BTreeMap<Abstraction, Histogram>>()
            .into()
    }
}

impl Save for Layer {
    fn name() -> &'static str {
        unreachable!("save lookups and transitions and metrics, not higher level layer")
    }
    fn done(street: Street) -> bool {
        Lookup::done(street) && Decomp::done(street) && Metric::done(street)
    }
    fn load(street: Street) -> Self {
        log::info!("{:<32}{:<32}", "loading layer", street);
        match street {
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
        }
    }
    fn save(&self) {
        log::info!("{:<32}{:<32}", "saving layer", self.street());
        self.metric().save();
        self.lookup().save();
        self.decomp().save();
    }
    fn make(street: Street) -> Self {
        log::info!("{:<32}{:<32}", "making layer", street);
        Self::load(street).cluster()
    }
}
