//! Definition syntax: tokenization, recursive-descent parsing and declaration AST.
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

/// Source declarations; members retain their owner instead of becoming flat records.
#[derive(Debug, Clone)]
pub struct Decl {
    pub offset: usize,
    pub name: String,
    pub fields: BTreeMap<String, Node>,
    pub kind: DeclKind,
}

#[derive(Debug, Clone)]
pub enum DeclKind {
    Type {
        binding: Node,
        members: Vec<Decl>,
    },
    TypeSet(Node),
    Function {
        signature: Signature,
        body: FunctionBody,
    },
    Constant(Node),
    Op(Signature),
    /// Declarations whose body is entirely described by a field schema.
    Fields(String),
}

impl Decl {
    pub fn tag(&self) -> &str {
        match &self.kind {
            DeclKind::Type { .. } => "type",
            DeclKind::TypeSet(_) => "typeset",
            DeclKind::Function { .. } => "fn",
            DeclKind::Constant(_) => "const",
            DeclKind::Op(_) => "op",
            DeclKind::Fields(name) => name,
        }
    }

    pub fn signature(&self) -> Option<&Signature> {
        match &self.kind {
            DeclKind::Function { signature, .. } | DeclKind::Op(signature) => Some(signature),
            _ => None,
        }
    }

    pub fn body(&self) -> Option<&FunctionBody> {
        match &self.kind {
            DeclKind::Function { body, .. } => Some(body),
            _ => None,
        }
    }

    pub fn members(&self) -> &[Decl] {
        match &self.kind {
            DeclKind::Type { members, .. } => members,
            _ => &[],
        }
    }

    /// Relocate an independently parsed file, including nested members.
    pub(crate) fn relocate(&mut self, base: usize) {
        self.offset += base;
        for node in self.fields.values_mut() {
            node.relocate(base);
        }
        match &mut self.kind {
            DeclKind::Type { binding, members } => {
                binding.relocate(base);
                for member in members {
                    member.relocate(base);
                }
            }
            DeclKind::TypeSet(node) | DeclKind::Constant(node) => node.relocate(base),
            DeclKind::Function { signature, body } => {
                signature.relocate(base);
                match body {
                    FunctionBody::Value(node) => node.relocate(base),
                    FunctionBody::Rust { offset, .. } => *offset += base,
                }
            }
            DeclKind::Op(signature) => signature.relocate(base),
            DeclKind::Fields(_) => {}
        }
    }
}

/// Visit declarations and their members without creating a flattened AST.
pub fn walk(declarations: &[Decl]) -> impl Iterator<Item = (Option<&str>, &Decl)> {
    declarations.iter().flat_map(|decl| {
        std::iter::once((None, decl)).chain(
            decl.members()
                .iter()
                .map(move |member| (Some(decl.name.as_str()), member)),
        )
    })
}

impl Signature {
    fn relocate(&mut self, base: usize) {
        for param in self.generics.iter_mut().chain(&mut self.params) {
            param.offset += base;
            param.ty.relocate(base);
        }
        if let Results::Fixed(results) = &mut self.results {
            for result in results {
                result.offset += base;
                result.ty.relocate(base);
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
    pub declarations: Vec<Decl>,
}

pub struct Import {
    pub offset: usize,
    pub path: String,
}
