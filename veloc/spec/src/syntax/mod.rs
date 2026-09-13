//! Definition syntax: a streaming lexer, recursive-descent parser and untyped AST.
//! Meaning and cross-declaration checks belong to the checked model.
mod lexer;
mod parser;

pub use parser::parse;
pub(crate) use parser::parse_file;

use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct Node {
    pub offset: usize,
    pub kind: Kind,
}

#[derive(Debug, Clone)]
pub enum Kind {
    Name(String),
    Text(String),
    Number(u32),
    List(Vec<Node>),
    Call(String, Vec<Node>),
    Member(Box<Node>, String),
    Method(Box<Node>, String, Vec<Node>),
    Object(String, BTreeMap<String, Node>),
    Union(Vec<Node>),
    Intersection(Vec<Node>),
    Integer(i128),
    Unary(&'static str, Box<Node>),
    Binary(&'static str, Box<Node>, Box<Node>),
    Lambda(Vec<String>, Box<Node>),
    Try(Box<Node>),
    Ref(Box<Node>),
    /// A field body with an explicit, typed host context.
    Scoped(Box<Parameter>, Box<Node>),
    Let(String, Box<Node>),
    Query(String, Box<Node>),
}

#[derive(Debug, Clone)]
pub struct Parameter {
    pub offset: usize,
    pub name: String,
    pub moves: bool,
    pub ty: Node,
}

#[derive(Debug, Clone)]
pub struct ResultType {
    pub offset: usize,
    pub name: Option<String>,
    pub ty: Node,
}

#[derive(Debug, Clone)]
pub enum Results {
    Fixed(Vec<ResultType>),
    Signature,
}

#[derive(Debug, Clone)]
pub struct Signature {
    pub is_const: bool,
    pub generics: Vec<Parameter>,
    pub params: Vec<Parameter>,
    pub results: Results,
}

#[derive(Debug, Clone)]
pub enum FunctionBody {
    Value(Node),
    Rust { offset: usize, path: Option<String> },
}

#[derive(Debug, Clone)]
pub struct Record {
    pub offset: usize,
    pub kind: String,
    pub name: String,
    pub fields: BTreeMap<String, Node>,
    pub signature: Option<Signature>,
    pub body: Option<FunctionBody>,
}

impl Record {
    /// Relocate an independently parsed file into the compilation's source map.
    pub(crate) fn relocate(&mut self, base: usize) {
        self.offset += base;
        for node in self.fields.values_mut() {
            node.relocate(base);
        }
        match &mut self.body {
            Some(FunctionBody::Value(node)) => node.relocate(base),
            Some(FunctionBody::Rust { offset, .. }) => *offset += base,
            None => {}
        }
        if let Some(signature) = &mut self.signature {
            for param in signature.generics.iter_mut().chain(&mut signature.params) {
                param.offset += base;
                param.ty.relocate(base);
            }
            if let Results::Fixed(results) = &mut signature.results {
                for result in results {
                    result.offset += base;
                    result.ty.relocate(base);
                }
            }
        }
    }
}

impl Node {
    fn relocate(&mut self, base: usize) {
        self.offset += base;
        match &mut self.kind {
            Kind::List(nodes)
            | Kind::Call(_, nodes)
            | Kind::Union(nodes)
            | Kind::Intersection(nodes) => {
                for node in nodes {
                    node.relocate(base);
                }
            }
            Kind::Method(receiver, _, args) => {
                receiver.relocate(base);
                for arg in args {
                    arg.relocate(base);
                }
            }
            Kind::Object(_, fields) => {
                for node in fields.values_mut() {
                    node.relocate(base);
                }
            }
            Kind::Member(node, _)
            | Kind::Let(_, node)
            | Kind::Ref(node)
            | Kind::Unary(_, node)
            | Kind::Lambda(_, node)
            | Kind::Query(_, node)
            | Kind::Try(node) => node.relocate(base),
            Kind::Binary(_, lhs, rhs) => {
                lhs.relocate(base);
                rhs.relocate(base);
            }
            Kind::Scoped(param, body) => {
                param.offset += base;
                param.ty.relocate(base);
                body.relocate(base);
            }
            Kind::Name(_) | Kind::Text(_) | Kind::Number(_) | Kind::Integer(_) => {}
        }
    }
}

/// A file is parsed once, before the loader resolves its dependencies.
pub struct File {
    pub imports: Vec<Import>,
    pub records: Vec<Record>,
}

pub struct Import {
    pub offset: usize,
    pub path: String,
}
