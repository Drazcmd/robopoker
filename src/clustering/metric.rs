use super::equity::Equity;
use super::sinkhorn::Sinkhorn;
use crate::cards::street::Street;
use crate::clustering::abstraction::Abstraction;
use crate::clustering::histogram::Histogram;
use crate::clustering::pair::Pair;
use crate::transport::coupling::Coupling;
use crate::transport::measure::Measure;
use crate::Energy;
use crate::Save;
use std::collections::BTreeMap;

/// Distance metric for kmeans clustering.
/// encapsulates distance between `Abstraction`s of the "previous" hierarchy,
/// as well as: distance between `Histogram`s of the "current" hierarchy.
#[derive(Default)]
pub struct Metric(BTreeMap<Pair, Energy>);

impl Metric {
    const SUFFIX: &'static str = ".metric.pgcopy";

    fn lookup(&self, x: &Abstraction, y: &Abstraction) -> Energy {
        if x == y {
            0.
        } else {
            self.0
                .get(&Pair::from((x, y)))
                .copied()
                .expect("missing abstraction pair")
        }
    }

    pub fn emd(&self, source: &Histogram, target: &Histogram) -> Energy {
        match source.peek() {
            Abstraction::Learned(_) => Sinkhorn::from((source, target, self)).minimize().cost(),
            Abstraction::Percent(_) => Equity::variation(source, target),
            Abstraction::Preflop(_) => unreachable!("no preflop emd"),
        }
    }
    pub fn read() -> Self {
        log::info!("loading metric");
        Self(
            Street::all()
                .iter()
                .filter(|street| Self::done(**street))
                .fold(BTreeMap::default(), |mut map, street| {
                    map.extend(Self::load(*street).0);
                    map
                }),
        )
    }

    /// we're assuming tht the street is being generated AFTER the learned kmeans
    /// cluster distance calculation. so we should have (Street::K() choose 2)
    /// entreis in our abstraction pair lookup table.
    /// if this is off by just a few then it probably means a bunch of collisions
    /// maybe i should determinsitcally seed kmeans process, could be cool for reproducability too
    fn street(&self) -> Street {
        match self.0.len() {
            0 => Street::Rive,
            n if n == Street::Turn.k() => Street::Turn,
            n if n == Street::Flop.k() => Street::Flop,
            n if n == Street::Pref.k() => Street::Pref,
            n => panic!("incorrect N = {} entries in metric", n),
        }
    }
}

impl crate::Save for Metric {
    fn done(street: Street) -> bool {
        std::fs::metadata(format!("{}{}", street, Self::SUFFIX)).is_ok()
    }
    fn make(street: Street) -> Self {
        unreachable!("you have no business being calculated from scratch, rather than from default {street} ")
    }
    fn load(street: Street) -> Self {
        use byteorder::ReadBytesExt;
        use byteorder::BE;
        use std::fs::File;
        use std::io::BufReader;
        use std::io::Read;
        use std::io::Seek;
        use std::io::SeekFrom;
        let file = File::open(format!("{}{}", street, Self::SUFFIX)).expect("open file");
        let mut buffer = [0u8; 2];
        let mut lookup = BTreeMap::new();
        let mut reader = BufReader::new(file);
        reader.seek(SeekFrom::Start(19)).expect("seek past header");
        while reader.read_exact(&mut buffer).is_ok() {
            if u16::from_be_bytes(buffer) == 2 {
                reader.read_u32::<BE>().expect("pair length");
                let pair_i64 = reader.read_i64::<BE>().expect("read pair");
                reader.read_u32::<BE>().expect("distance length");
                let dist_f32 = reader.read_f32::<BE>().expect("read distance");
                let pair = Pair::from(pair_i64);
                lookup.insert(pair, dist_f32);
                continue;
            } else {
                break;
            }
        }
        Self(lookup)
    }
    fn save(&self) {
        let street = self.street();
        log::info!("{:<32}{:<32}", "saving metric", street);
        use byteorder::WriteBytesExt;
        use byteorder::BE;
        use std::fs::File;
        use std::io::Write;
        let ref mut file = File::create(format!("{}{}", street, Self::SUFFIX)).expect("touch");
        file.write_all(b"PGCOPY\n\xFF\r\n\0").expect("header");
        file.write_u32::<BE>(0).expect("flags");
        file.write_u32::<BE>(0).expect("extension");
        for (pair, distance) in self.0.iter() {
            const N_FIELDS: u16 = 2;
            file.write_u16::<BE>(N_FIELDS).unwrap();
            file.write_u32::<BE>(size_of::<i64>() as u32).unwrap();
            file.write_i64::<BE>(i64::from(*pair)).unwrap();
            file.write_u32::<BE>(size_of::<f32>() as u32).unwrap();
            file.write_f32::<BE>(*distance).unwrap();
        }
        file.write_u16::<BE>(0xFFFF).expect("trailer");
    }
}
impl Measure for Metric {
    type X = Abstraction;
    type Y = Abstraction;
    fn distance(&self, x: &Self::X, y: &Self::Y) -> Energy {
        match (x, y) {
            (Self::X::Learned(_), Self::Y::Learned(_)) => self.lookup(x, y),
            (Self::X::Percent(_), Self::Y::Percent(_)) => Equity.distance(x, y),
            (Self::X::Preflop(_), Self::Y::Preflop(_)) => unreachable!("no preflop distance"),
            _ => unreachable!(),
        }
    }
}

impl From<BTreeMap<Pair, Energy>> for Metric {
    fn from(metric: BTreeMap<Pair, Energy>) -> Self {
        let max = metric.values().copied().fold(f32::MIN_POSITIVE, f32::max);
        Self(
            metric
                .into_iter()
                .map(|(index, distance)| (index, distance / max))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::street::Street;
    use crate::clustering::emd::EMD;
    use crate::{Arbitrary, Save};

    #[test]
    fn persistence() {
        let street = Street::Rive;
        let emd = EMD::random();
        let save = emd.metric();
        save.save();
        let load = Metric::load(street);
        std::iter::empty()
            .chain(save.0.iter().zip(load.0.iter()))
            .chain(load.0.iter().zip(save.0.iter()))
            .all(|((s1, l1), (s2, l2))| s1 == s2 && l1 == l2);
    }
}
