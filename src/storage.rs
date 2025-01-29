use crate::rl::RLer;
use ron::de::from_reader;
use ron::ser::to_writer;
use std::fs::File;

pub use ron::error::{Result, SpannedResult};

pub fn save(model: RLer, filename: &str) -> Result<()> {
    let file = File::create(filename)?;
    to_writer(file, &model)
}

pub fn load(filename: &str) -> SpannedResult<RLer> {
    let file = File::open(filename)?;
    from_reader(file)
}
