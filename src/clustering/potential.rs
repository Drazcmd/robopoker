use super::abstraction::Abstraction;
use crate::transport::density::Density;
use crate::Distance;
use crate::Probability;
use std::collections::BTreeMap;

/// using this to represent an arbitrary instance of the Kontorovich-Rubinstein
/// potential formulation of the optimal transport problem.
/// this structure can also be treated as a normalized distribution over Abstractions.
pub struct Potential(BTreeMap<Abstraction, Probability>);

impl Potential {
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&Abstraction, &mut Probability)> {
        self.0.iter_mut()
    }
    pub fn values(&self) -> impl Iterator<Item = &Probability> {
        self.0.values()
    }
}

impl Density for Potential {
    type S = Abstraction;

    fn density(&self, x: &Self::S) -> Distance {
        self.0.get(x).copied().unwrap_or(0.)
    }
    fn support(&self) -> impl Iterator<Item = &Self::S> {
        self.0.keys()
    }
}

impl From<BTreeMap<Abstraction, Probability>> for Potential {
    fn from(potential: BTreeMap<Abstraction, Probability>) -> Self {
        Self(potential)
    }
}
