mod codec;
mod format;
pub mod parser;
pub mod printer;
pub(crate) use codec::TextCodec;

pub use parser::{ModuleParser, ParseError};
pub use printer::{FuncPrinter, InstPrinter, ModulePrinter};
