//! Finite comparison semantics. Trusted outcome transforms run at build time;
//! generated MIR APIs contain only direct matches over condition codes.

use std::collections::BTreeSet;
use std::fmt::Write;

use crate::Error;
use crate::model::{Fields, identifier, list, name};
use crate::syntax::{Kind, Node, Record};

const LESS: u8 = 1;
const EQUAL: u8 = 2;
const GREATER: u8 = 4;
const UNORDERED: u8 = 8;
const ORDERED: u8 = LESS | EQUAL | GREATER;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Domain {
    Integer,
    Float,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Order {
    Independent,
    Signed,
    Unsigned,
}

struct Predicate {
    offset: usize,
    name: String,
    mnemonic: String,
    order: Order,
    outcomes: u8,
}

pub(crate) struct Comparison {
    pub(crate) name: String,
    domain: Domain,
    predicates: Vec<Predicate>,
    swap: Vec<usize>,
    complement: Vec<Option<usize>>,
    ordered_complement: Vec<usize>,
}

pub(crate) fn compile(records: &[Record], source: &str) -> Result<Vec<Comparison>, Error> {
    records
        .iter()
        .filter(|r| r.kind == "comparison")
        .map(|r| Comparison::compile(r, source))
        .collect()
}

impl Comparison {
    pub(crate) fn has_variant(&self, name: &str) -> bool {
        self.predicates
            .iter()
            .any(|predicate| predicate.name == name)
    }

    fn compile(record: &Record, source: &str) -> Result<Self, Error> {
        let mut fields = Fields::new(source, record.clone());
        let node = fields.take("domain")?;
        let domain = match name(source, node.clone())?.as_str() {
            "integer" => Domain::Integer,
            "float" => Domain::Float,
            _ => {
                return Err(Error::at(
                    source,
                    node.offset,
                    "expected integer or float domain",
                ));
            }
        };
        let mut predicates = Vec::<Predicate>::new();
        let mut names = BTreeSet::new();
        for node in list(source, fields.take("predicates")?)? {
            let predicate = Predicate::parse(node, domain, source)?;
            if !names.insert(predicate.mnemonic.clone()) {
                return Err(Error::at(
                    source,
                    predicate.offset,
                    "duplicate comparison mnemonic",
                ));
            }
            if predicates
                .iter()
                .any(|p| p.order == predicate.order && p.outcomes == predicate.outcomes)
            {
                return Err(Error::at(
                    source,
                    predicate.offset,
                    "duplicate comparison semantics",
                ));
            }
            predicates.push(predicate);
        }
        fields.finish()?;
        if predicates.is_empty() {
            return Err(fields.error("comparison must declare at least one predicate"));
        }
        let mut comparison = Self {
            name: record.name.clone(),
            domain,
            predicates,
            swap: Vec::new(),
            complement: Vec::new(),
            ordered_complement: Vec::new(),
        };
        let universe = if domain == Domain::Float {
            ORDERED | UNORDERED
        } else {
            ORDERED
        };
        for p in &comparison.predicates {
            let required = |target: Option<usize>, transform: &str| {
                target.ok_or_else(|| {
                    Error::at(
                        source,
                        p.offset,
                        format!(
                            "predicate `{}` has no {transform} in `{}`",
                            p.name, comparison.name
                        ),
                    )
                })
            };
            let swapped = (p.outcomes & !(LESS | GREATER))
                | ((p.outcomes & LESS) << 2)
                | ((p.outcomes & GREATER) >> 2);
            comparison.swap.push(required(
                comparison.find(p.order, swapped, universe),
                "swap",
            )?);
            let complement = comparison.find(p.order, p.outcomes ^ universe, universe);
            if domain == Domain::Integer {
                required(complement, "complement")?;
            }
            comparison.complement.push(complement);
            if domain == Domain::Float {
                comparison.ordered_complement.push(required(
                    comparison.find(p.order, (p.outcomes & ORDERED) ^ ORDERED, ORDERED),
                    "ordered complement",
                )?);
            }
        }
        Ok(comparison)
    }

    fn find(&self, order: Order, outcomes: u8, universe: u8) -> Option<usize> {
        // Under a restricted domain multiple predicates may be equivalent.
        // Prefer the exact outcome set, then the first declared equivalent.
        let matches = |p: &Predicate| {
            (p.order == order || p.order == Order::Independent) && p.outcomes & universe == outcomes
        };
        self.predicates
            .iter()
            .position(|p| matches(p) && p.outcomes == outcomes)
            .or_else(|| self.predicates.iter().position(matches))
    }

    pub(crate) fn generate(&self) -> String {
        let mut out = String::new();
        writeln!(
            out,
            "#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum {} {{",
            self.name
        )
        .unwrap();
        for p in &self.predicates {
            writeln!(out, "    {},", p.name).unwrap();
        }
        writeln!(out, "}}\nimpl {} {{", self.name).unwrap();
        self.method(
            &mut out,
            "mnemonic",
            "&'static str",
            self.predicates.iter().map(|p| format!("{:?}", p.mnemonic)),
        );
        out.push_str("pub fn from_mnemonic(mnemonic: &str) -> Option<Self> { match mnemonic {\n");
        for p in &self.predicates {
            writeln!(out, "{:?} => Some(Self::{}),", p.mnemonic, p.name).unwrap();
        }
        out.push_str("_ => None,\n} }\n");
        if self.domain == Domain::Integer {
            self.method(
                &mut out,
                "predicate",
                "crate::semantics::IntPredicate",
                self.predicates.iter().map(|p| {
                    format!(
                        "crate::semantics::IntPredicate::new({}, {})",
                        p.order == Order::Signed,
                        p.outcomes
                    )
                }),
            );
            self.method(
                &mut out,
                "is_unsigned",
                "bool",
                self.predicates
                    .iter()
                    .map(|p| (p.order == Order::Unsigned).to_string()),
            );
        }
        let variant = |index: usize| format!("Self::{}", self.predicates[index].name);
        self.method(
            &mut out,
            "swap",
            "Self",
            self.swap.iter().map(|&i| variant(i)),
        );
        if self.domain == Domain::Integer {
            self.method(
                &mut out,
                "complement",
                "Self",
                self.complement
                    .iter()
                    .map(|i| variant(i.expect("checked integer complement"))),
            );
        } else {
            out.push_str("/// Exact IEEE logical complement, or None if it is not represented.\n");
            self.method(
                &mut out,
                "complement",
                "Option<Self>",
                self.complement
                    .iter()
                    .map(|i| i.map_or_else(|| "None".into(), |i| format!("Some({})", variant(i)))),
            );
            out.push_str("/// Complement valid only when both operands are known not to be NaN.\n");
            self.method(
                &mut out,
                "complement_ordered",
                "Self",
                self.ordered_complement.iter().map(|&i| variant(i)),
            );
        }
        writeln!(out, "}}\nimpl core::fmt::Display for {} {{\nfn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {{ f.write_str(self.mnemonic()) }}\n}}", self.name).unwrap();
        out
    }

    fn method(
        &self,
        out: &mut String,
        name: &str,
        result: &str,
        arms: impl Iterator<Item = String>,
    ) {
        writeln!(
            out,
            "pub const fn {name}(self) -> {result} {{ match self {{"
        )
        .unwrap();
        for (p, arm) in self.predicates.iter().zip(arms) {
            writeln!(out, "Self::{} => {arm},", p.name).unwrap();
        }
        out.push_str("} }\n");
    }
}

impl Predicate {
    fn parse(node: Node, domain: Domain, source: &str) -> Result<Self, Error> {
        let fail = |message| Error::at(source, node.offset, message);
        let Kind::Call(variant, args) = node.kind else {
            return Err(fail(
                "expected Variant([outcomes]) or Variant(order, [outcomes])",
            ));
        };
        identifier(source, node.offset, &variant)?;
        let (order, outcomes) = match args.as_slice() {
            [outcomes] => (Order::Independent, outcomes),
            [order, outcomes] if domain == Domain::Integer => {
                let order = match name(source, order.clone())?.as_str() {
                    "signed" => Order::Signed,
                    "unsigned" => Order::Unsigned,
                    _ => return Err(fail("expected signed or unsigned order")),
                };
                (order, outcomes)
            }
            _ => {
                return Err(fail(
                    "expected an outcome list, optionally preceded by integer signedness",
                ));
            }
        };
        let mut bits = 0;
        for outcome in list(source, outcomes.clone())? {
            let bit = match name(source, outcome.clone())?.as_str() {
                "less" => LESS,
                "equal" => EQUAL,
                "greater" => GREATER,
                "unordered" if domain == Domain::Float => UNORDERED,
                _ => {
                    return Err(Error::at(
                        source,
                        outcome.offset,
                        "invalid comparison outcome for this domain",
                    ));
                }
            };
            if bits & bit != 0 {
                return Err(Error::at(
                    source,
                    outcome.offset,
                    "duplicate comparison outcome",
                ));
            }
            bits |= bit;
        }
        if domain == Domain::Integer {
            let independent = (bits & LESS != 0) == (bits & GREATER != 0);
            if independent != (order == Order::Independent) {
                return Err(fail(
                    "integer ordering requires signedness; equality predicates must omit it",
                ));
            }
        }
        Ok(Self {
            offset: node.offset,
            mnemonic: variant.to_ascii_lowercase(),
            name: variant,
            order,
            outcomes: bits,
        })
    }
}
