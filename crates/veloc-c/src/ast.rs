//! Abstract Syntax Tree for C Language
//!
//! This module defines the AST nodes for C99/C11 language.

#![allow(dead_code)]

use std::boxed::Box;
use std::string::String;
use std::vec::Vec;

/// A translation unit (source file)
#[derive(Debug, Clone, PartialEq)]
pub struct TranslationUnit {
    pub declarations: Vec<ExternalDeclaration>,
}

impl TranslationUnit {
    pub fn new(declarations: Vec<ExternalDeclaration>) -> Self {
        TranslationUnit { declarations }
    }
}

/// External declaration (top-level)
#[derive(Debug, Clone, PartialEq)]
pub enum ExternalDeclaration {
    FunctionDefinition(FunctionDefinition),
    Declaration(Declaration),
}

/// Function definition
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDefinition {
    pub specifiers: Vec<DeclarationSpecifier>,
    pub declarator: Declarator,
    pub declarations: Vec<Declaration>,
    pub body: CompoundStatement,
}

impl FunctionDefinition {
    pub fn new(
        specifiers: Vec<DeclarationSpecifier>,
        declarator: Declarator,
        declarations: Vec<Declaration>,
        body: CompoundStatement,
    ) -> Self {
        FunctionDefinition {
            specifiers,
            declarator,
            declarations,
            body,
        }
    }
}

/// Declaration
#[derive(Debug, Clone, PartialEq)]
pub struct Declaration {
    pub specifiers: Vec<DeclarationSpecifier>,
    pub init_declarators: Vec<InitDeclarator>,
}

impl Declaration {
    pub fn new(
        specifiers: Vec<DeclarationSpecifier>,
        init_declarators: Vec<InitDeclarator>,
    ) -> Self {
        Declaration {
            specifiers,
            init_declarators,
        }
    }
}

/// Declaration specifier
#[derive(Debug, Clone, PartialEq)]
pub enum DeclarationSpecifier {
    StorageClass(StorageClassSpecifier),
    TypeSpecifier(TypeSpecifier),
    TypeQualifier(TypeQualifier),
    FunctionSpecifier(FunctionSpecifier),
}

/// Storage class specifier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageClassSpecifier {
    Typedef,
    Extern,
    Static,
    Auto,
    Register,
}

/// Type specifier
#[derive(Debug, Clone, PartialEq)]
pub enum TypeSpecifier {
    Void,
    Char,
    Short,
    Int,
    Long,
    Float,
    Double,
    Signed,
    Unsigned,
    Bool,
    Complex,
    Imaginary,
    Struct(StructSpecifier),
    Enum(EnumSpecifier),
    TypedefName(String),
}

/// Struct or union specifier
#[derive(Debug, Clone, PartialEq)]
pub struct StructSpecifier {
    pub is_union: bool,
    pub name: Option<String>,
    pub members: Option<Vec<StructDeclaration>>,
}

impl StructSpecifier {
    pub fn new_struct(name: Option<String>, members: Option<Vec<StructDeclaration>>) -> Self {
        StructSpecifier {
            is_union: false,
            name,
            members,
        }
    }

    pub fn new_union(name: Option<String>, members: Option<Vec<StructDeclaration>>) -> Self {
        StructSpecifier {
            is_union: true,
            name,
            members,
        }
    }
}

/// Struct declaration (member declaration)
#[derive(Debug, Clone, PartialEq)]
pub struct StructDeclaration {
    pub specifiers: Vec<SpecifierQualifier>,
    pub declarators: Vec<StructDeclarator>,
}

/// Specifier qualifier
#[derive(Debug, Clone, PartialEq)]
pub enum SpecifierQualifier {
    TypeSpecifier(TypeSpecifier),
    TypeQualifier(TypeQualifier),
}

/// Struct declarator
#[derive(Debug, Clone, PartialEq)]
pub struct StructDeclarator {
    pub declarator: Option<Declarator>,
    pub bit_width: Option<Expression>,
}

/// Enum specifier
#[derive(Debug, Clone, PartialEq)]
pub struct EnumSpecifier {
    pub name: Option<String>,
    pub enumerators: Option<Vec<Enumerator>>,
}

/// Enumerator
#[derive(Debug, Clone, PartialEq)]
pub struct Enumerator {
    pub name: String,
    pub value: Option<Expression>,
}

/// Type qualifier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeQualifier {
    Const,
    Restrict,
    Volatile,
}

/// Function specifier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FunctionSpecifier {
    Inline,
}

/// Declarator with optional initializer
#[derive(Debug, Clone, PartialEq)]
pub struct InitDeclarator {
    pub declarator: Declarator,
    pub initializer: Option<Initializer>,
}

/// Declarator
#[derive(Debug, Clone, PartialEq)]
pub struct Declarator {
    pub pointer: Option<Pointer>,
    pub direct: DirectDeclarator,
}

impl Declarator {
    pub fn new(pointer: Option<Pointer>, direct: DirectDeclarator) -> Self {
        Declarator { pointer, direct }
    }

    /// Get the name of the declarator
    pub fn name(&self) -> &str {
        self.direct.name()
    }
}

/// Direct declarator
#[derive(Debug, Clone, PartialEq)]
pub enum DirectDeclarator {
    Identifier(String),
    Parenthesized(Box<Declarator>),
    Array(Box<DirectDeclarator>, Option<Expression>),
    Function(Box<DirectDeclarator>, Option<Vec<ParameterDeclaration>>),
    FunctionOldStyle(Box<DirectDeclarator>, Option<Vec<String>>),
}

impl DirectDeclarator {
    /// Get the name of the direct declarator
    pub fn name(&self) -> &str {
        match self {
            DirectDeclarator::Identifier(name) => name,
            DirectDeclarator::Parenthesized(decl) => decl.name(),
            DirectDeclarator::Array(inner, _) => inner.name(),
            DirectDeclarator::Function(inner, _) => inner.name(),
            DirectDeclarator::FunctionOldStyle(inner, _) => inner.name(),
        }
    }
}

/// Pointer
#[derive(Debug, Clone, PartialEq)]
pub struct Pointer {
    pub qualifiers: Vec<TypeQualifier>,
    pub inner: Option<Box<Pointer>>,
}

impl Pointer {
    pub fn new(qualifiers: Vec<TypeQualifier>, inner: Option<Box<Pointer>>) -> Self {
        Pointer { qualifiers, inner }
    }
}

/// Parameter declaration
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterDeclaration {
    pub specifiers: Vec<DeclarationSpecifier>,
    pub declarator: Option<Declarator>,
    pub abstract_declarator: Option<AbstractDeclarator>,
}

/// Abstract declarator (for type names without identifier)
#[derive(Debug, Clone, PartialEq)]
pub struct AbstractDeclarator {
    pub pointer: Option<Pointer>,
    pub direct: Option<Box<DirectAbstractDeclarator>>,
}

/// Direct abstract declarator
#[derive(Debug, Clone, PartialEq)]
pub enum DirectAbstractDeclarator {
    Parenthesized(Box<AbstractDeclarator>),
    Array(
        Option<Box<DirectAbstractDeclarator>>,
        Option<Box<Expression>>,
    ),
    Function(
        Box<Option<DirectAbstractDeclarator>>,
        Option<Vec<ParameterDeclaration>>,
    ),
}

/// Initializer
#[derive(Debug, Clone, PartialEq)]
pub enum Initializer {
    Expression(Expression),
}

/// Designator
#[derive(Debug, Clone, PartialEq)]
pub enum Designator {
    ArrayIndex(Expression),
    Member(String),
}

// ========== Statements ==========

/// Statement
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Labeled(LabeledStatement),
    Compound(CompoundStatement),
    Expression(ExpressionStatement),
    Selection(SelectionStatement),
    Iteration(IterationStatement),
    Jump(JumpStatement),
}

/// Labeled statement
#[derive(Debug, Clone, PartialEq)]
pub enum LabeledStatement {
    Label(String, Box<Statement>),
    Case(Expression, Box<Statement>),
    Default(Box<Statement>),
}

/// Compound statement (block)
#[derive(Debug, Clone, PartialEq)]
pub struct CompoundStatement {
    pub items: Vec<BlockItem>,
}

impl CompoundStatement {
    pub fn new(items: Vec<BlockItem>) -> Self {
        CompoundStatement { items }
    }

    pub fn empty() -> Self {
        CompoundStatement { items: Vec::new() }
    }
}

/// Block item
#[derive(Debug, Clone, PartialEq)]
pub enum BlockItem {
    Declaration(Declaration),
    Statement(Statement),
}

/// Expression statement
#[derive(Debug, Clone, PartialEq)]
pub struct ExpressionStatement {
    pub expression: Option<Expression>,
}

/// Selection statement
#[derive(Debug, Clone, PartialEq)]
pub enum SelectionStatement {
    If(Expression, Box<Statement>, Option<Box<Statement>>),
    Switch(Expression, Box<Statement>),
}

/// Iteration statement
#[derive(Debug, Clone, PartialEq)]
pub enum IterationStatement {
    While(Expression, Box<Statement>),
    DoWhile(Box<Statement>, Expression),
    For(
        ForInit,
        Option<Expression>,
        Option<Expression>,
        Box<Statement>,
    ),
}

/// For loop initialization
#[derive(Debug, Clone, PartialEq)]
pub enum ForInit {
    Expression(Option<Expression>),
    Declaration(Declaration),
}

/// Jump statement
#[derive(Debug, Clone, PartialEq)]
pub enum JumpStatement {
    Goto(String),
    Continue,
    Break,
    Return(Option<Expression>),
}

// ========== Expressions ==========

/// Expression
#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    /// Primary expressions
    Identifier(String),
    Integer(i64),
    UnsignedInteger(u64),
    Float(f64),
    Char(char),
    String(String),
    Parenthesized(Box<Expression>),

    /// Generic selection (C11)
    Generic(Box<Expression>, Vec<GenericAssociation>),

    /// Postfix expressions
    ArrayAccess(Box<Expression>, Box<Expression>),
    FunctionCall(Box<Expression>, Vec<Expression>),
    MemberAccess(Box<Expression>, String),
    PointerMemberAccess(Box<Expression>, String),
    PostIncrement(Box<Expression>),
    PostDecrement(Box<Expression>),
    CompoundLiteral(Vec<SpecifierQualifier>, Box<Expression>),

    /// Unary expressions
    PreIncrement(Box<Expression>),
    PreDecrement(Box<Expression>),
    AddressOf(Box<Expression>),
    Dereference(Box<Expression>),
    UnaryPlus(Box<Expression>),
    UnaryMinus(Box<Expression>),
    BitwiseNot(Box<Expression>),
    LogicalNot(Box<Expression>),
    SizeofExpression(Box<Expression>),
    SizeofType(Vec<SpecifierQualifier>, Option<Box<AbstractDeclarator>>),
    AlignofType(Vec<SpecifierQualifier>, Option<AbstractDeclarator>),

    /// Cast expression
    Cast(TypeName, Box<Expression>),

    /// Binary expressions
    Multiply(Box<Expression>, Box<Expression>),
    Divide(Box<Expression>, Box<Expression>),
    Modulo(Box<Expression>, Box<Expression>),
    Add(Box<Expression>, Box<Expression>),
    Subtract(Box<Expression>, Box<Expression>),
    ShiftLeft(Box<Expression>, Box<Expression>),
    ShiftRight(Box<Expression>, Box<Expression>),
    LessThan(Box<Expression>, Box<Expression>),
    LessThanOrEqual(Box<Expression>, Box<Expression>),
    GreaterThan(Box<Expression>, Box<Expression>),
    GreaterThanOrEqual(Box<Expression>, Box<Expression>),
    Equal(Box<Expression>, Box<Expression>),
    NotEqual(Box<Expression>, Box<Expression>),
    BitwiseAnd(Box<Expression>, Box<Expression>),
    BitwiseXor(Box<Expression>, Box<Expression>),
    BitwiseOr(Box<Expression>, Box<Expression>),
    LogicalAnd(Box<Expression>, Box<Expression>),
    LogicalOr(Box<Expression>, Box<Expression>),

    /// Conditional expression
    Conditional(Box<Expression>, Box<Expression>, Box<Expression>),

    /// Assignment expressions
    Assign(Box<Expression>, Box<Expression>),
    MultiplyAssign(Box<Expression>, Box<Expression>),
    DivideAssign(Box<Expression>, Box<Expression>),
    ModuloAssign(Box<Expression>, Box<Expression>),
    AddAssign(Box<Expression>, Box<Expression>),
    SubtractAssign(Box<Expression>, Box<Expression>),
    ShiftLeftAssign(Box<Expression>, Box<Expression>),
    ShiftRightAssign(Box<Expression>, Box<Expression>),
    BitwiseAndAssign(Box<Expression>, Box<Expression>),
    BitwiseXorAssign(Box<Expression>, Box<Expression>),
    BitwiseOrAssign(Box<Expression>, Box<Expression>),

    /// Comma expression
    Comma(Box<Expression>, Box<Expression>),
}

/// Generic association
#[derive(Debug, Clone, PartialEq)]
pub enum GenericAssociation {
    Type(TypeName, Expression),
    Default(Expression),
}

/// Type name
#[derive(Debug, Clone, PartialEq)]
pub struct TypeName {
    pub specifiers: Vec<SpecifierQualifier>,
    pub abstract_declarator: Option<AbstractDeclarator>,
}
