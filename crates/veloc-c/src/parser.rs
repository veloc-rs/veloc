//! C Language Parser
//!
//! Recursive descent parser for C99/C11 language.

#![allow(dead_code)]

use crate::ast::*;
use crate::error::{Error, Result};
use crate::lexer::{Lexer, Token, TokenKind};
use std::boxed::Box;
use std::format;
use std::string::ToString;
use std::vec::Vec;

/// C Language Parser
pub struct Parser<'a> {
    lexer: Lexer<'a>,
    current: Option<Token>,
}

impl<'a> Parser<'a> {
    /// Create a new parser from a lexer
    pub fn new(mut lexer: Lexer<'a>) -> Self {
        let current = lexer.next_token().ok();
        Parser { lexer, current }
    }

    /// Parse a translation unit (source file)
    pub fn parse_translation_unit(&mut self) -> Result<TranslationUnit> {
        let mut declarations = Vec::new();

        while !self.is_at_end() {
            let decl = self.parse_external_declaration()?;
            declarations.push(decl);
        }

        Ok(TranslationUnit::new(declarations))
    }

    /// Check if we've reached the end of input
    fn is_at_end(&self) -> bool {
        matches!(&self.current, Some(t) if t.kind == TokenKind::Eof)
    }

    /// Get the current token
    fn current(&self) -> Result<&Token> {
        self.current
            .as_ref()
            .ok_or_else(|| Error::syntax("Unexpected end of input", 0, 0))
    }

    /// Advance to the next token
    fn advance(&mut self) -> Result<Token> {
        let prev = self
            .current
            .take()
            .ok_or_else(|| Error::syntax("Unexpected end of input", 0, 0))?;
        self.current = self.lexer.next_token().ok();
        Ok(prev)
    }

    /// Peek at the current token kind
    fn peek_kind(&self) -> Option<TokenKind> {
        self.current.as_ref().map(|t| t.kind)
    }

    /// Check if current token matches the expected kind
    fn check(&self, kind: TokenKind) -> bool {
        matches!(self.peek_kind(), Some(k) if k == kind)
    }

    /// Match and consume a token of the given kind
    fn match_token(&mut self, kind: TokenKind) -> Result<bool> {
        if self.check(kind) {
            self.advance()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Expect a specific token kind, error if not found
    fn expect(&mut self, kind: TokenKind, message: &str) -> Result<Token> {
        if self.check(kind) {
            self.advance()
        } else {
            let token = self.current()?;
            Err(Error::syntax(
                format!("{}: expected {:?}, got {:?}", message, kind, token.kind),
                token.line,
                token.column,
            ))
        }
    }

    /// Parse external declaration
    fn parse_external_declaration(&mut self) -> Result<ExternalDeclaration> {
        // Parse specifiers and declarator first
        let specifiers = self.parse_declaration_specifiers()?;
        let declarator = self.parse_declarator()?;

        // Check if this is a function definition (function declarator followed by {)
        if self.is_function_declarator(&declarator) && self.check(TokenKind::LBrace) {
            // This is a function definition
            let body = self.parse_compound_statement()?;
            Ok(ExternalDeclaration::FunctionDefinition(
                FunctionDefinition::new(specifiers, declarator, Vec::new(), body),
            ))
        } else {
            // This is a declaration (may have init declarators)
            let mut init_declarators = vec![InitDeclarator {
                declarator,
                initializer: None,
            }];

            // Check for initializer
            if self.match_token(TokenKind::Assign)? {
                init_declarators[0].initializer = Some(self.parse_initializer()?);
            }

            // Parse additional declarators if any
            while self.match_token(TokenKind::Comma)? {
                let declarator = self.parse_declarator()?;
                let initializer = if self.match_token(TokenKind::Assign)? {
                    Some(self.parse_initializer()?)
                } else {
                    None
                };
                init_declarators.push(InitDeclarator {
                    declarator,
                    initializer,
                });
            }

            self.expect(TokenKind::Semicolon, "Expected ';' after declaration")?;
            Ok(ExternalDeclaration::Declaration(Declaration::new(
                specifiers,
                init_declarators,
            )))
        }
    }

    /// Check if a declarator is a function declarator
    fn is_function_declarator(&self, declarator: &Declarator) -> bool {
        matches!(&declarator.direct, DirectDeclarator::Function(_, _))
    }

    /// Save parser state for backtracking
    fn save_checkpoint(&self) -> ParserCheckpoint {
        ParserCheckpoint {
            current: self.current.clone(),
        }
    }

    /// Restore parser state from checkpoint
    fn restore_checkpoint(&mut self, checkpoint: ParserCheckpoint) {
        self.current = checkpoint.current;
    }

    /// Parse declaration specifiers
    fn parse_declaration_specifiers(&mut self) -> Result<Vec<DeclarationSpecifier>> {
        let mut specifiers = Vec::new();

        loop {
            match self.peek_kind() {
                // Storage class specifiers
                Some(TokenKind::Typedef) => {
                    self.advance()?;
                    specifiers.push(DeclarationSpecifier::StorageClass(
                        StorageClassSpecifier::Typedef,
                    ));
                }
                Some(TokenKind::Extern) => {
                    self.advance()?;
                    specifiers.push(DeclarationSpecifier::StorageClass(
                        StorageClassSpecifier::Extern,
                    ));
                }
                Some(TokenKind::Static) => {
                    self.advance()?;
                    specifiers.push(DeclarationSpecifier::StorageClass(
                        StorageClassSpecifier::Static,
                    ));
                }
                Some(TokenKind::Auto) => {
                    self.advance()?;
                    specifiers.push(DeclarationSpecifier::StorageClass(
                        StorageClassSpecifier::Auto,
                    ));
                }
                Some(TokenKind::Register) => {
                    self.advance()?;
                    specifiers.push(DeclarationSpecifier::StorageClass(
                        StorageClassSpecifier::Register,
                    ));
                }
                // Type specifiers
                Some(TokenKind::Void) => {
                    self.advance()?;
                    specifiers.push(DeclarationSpecifier::TypeSpecifier(TypeSpecifier::Void));
                }
                Some(TokenKind::CharKw) => {
                    self.advance()?;
                    specifiers.push(DeclarationSpecifier::TypeSpecifier(TypeSpecifier::Char));
                }
                Some(TokenKind::Short) => {
                    self.advance()?;
                    specifiers.push(DeclarationSpecifier::TypeSpecifier(TypeSpecifier::Short));
                }
                Some(TokenKind::Int) => {
                    self.advance()?;
                    specifiers.push(DeclarationSpecifier::TypeSpecifier(TypeSpecifier::Int));
                }
                Some(TokenKind::Long) => {
                    self.advance()?;
                    specifiers.push(DeclarationSpecifier::TypeSpecifier(TypeSpecifier::Long));
                }
                Some(TokenKind::FloatKw) => {
                    self.advance()?;
                    specifiers.push(DeclarationSpecifier::TypeSpecifier(TypeSpecifier::Float));
                }
                Some(TokenKind::Double) => {
                    self.advance()?;
                    specifiers.push(DeclarationSpecifier::TypeSpecifier(TypeSpecifier::Double));
                }
                Some(TokenKind::Signed) => {
                    self.advance()?;
                    specifiers.push(DeclarationSpecifier::TypeSpecifier(TypeSpecifier::Signed));
                }
                Some(TokenKind::Unsigned) => {
                    self.advance()?;
                    specifiers.push(DeclarationSpecifier::TypeSpecifier(TypeSpecifier::Unsigned));
                }
                Some(TokenKind::Struct) => {
                    self.advance()?;
                    specifiers.push(DeclarationSpecifier::TypeSpecifier(TypeSpecifier::Struct(
                        self.parse_struct_specifier(false)?,
                    )));
                }
                Some(TokenKind::Union) => {
                    self.advance()?;
                    specifiers.push(DeclarationSpecifier::TypeSpecifier(TypeSpecifier::Struct(
                        self.parse_struct_specifier(true)?,
                    )));
                }
                Some(TokenKind::Enum) => {
                    self.advance()?;
                    specifiers.push(DeclarationSpecifier::TypeSpecifier(TypeSpecifier::Enum(
                        self.parse_enum_specifier()?,
                    )));
                }
                // Type qualifiers
                Some(TokenKind::Const) => {
                    self.advance()?;
                    specifiers.push(DeclarationSpecifier::TypeQualifier(TypeQualifier::Const));
                }
                Some(TokenKind::Restrict) => {
                    self.advance()?;
                    specifiers.push(DeclarationSpecifier::TypeQualifier(TypeQualifier::Restrict));
                }
                Some(TokenKind::Volatile) => {
                    self.advance()?;
                    specifiers.push(DeclarationSpecifier::TypeQualifier(TypeQualifier::Volatile));
                }
                // Function specifiers
                Some(TokenKind::Inline) => {
                    self.advance()?;
                    specifiers.push(DeclarationSpecifier::FunctionSpecifier(
                        FunctionSpecifier::Inline,
                    ));
                }
                // Identifiers (typedef names)
                Some(TokenKind::Identifier) => {
                    // For simplicity, we're treating all identifiers after specifiers as typedef names
                    // In a real implementation, you'd need a symbol table
                    break;
                }
                _ => break,
            }
        }

        if specifiers.is_empty() {
            let token = self.current()?;
            return Err(Error::syntax(
                "Expected declaration specifier",
                token.line,
                token.column,
            ));
        }

        Ok(specifiers)
    }

    /// Parse struct specifier
    fn parse_struct_specifier(&mut self, is_union: bool) -> Result<StructSpecifier> {
        let name = if let Some(TokenKind::Identifier) = self.peek_kind() {
            Some(self.advance()?.lexeme)
        } else {
            None
        };

        let members = if self.check(TokenKind::LBrace) {
            self.advance()?;
            let mut decls = Vec::new();
            while !self.check(TokenKind::RBrace) && !self.is_at_end() {
                decls.push(self.parse_struct_declaration()?);
            }
            self.expect(TokenKind::RBrace, "Expected '}' after struct members")?;
            Some(decls)
        } else {
            None
        };

        if is_union {
            Ok(StructSpecifier::new_union(name, members))
        } else {
            Ok(StructSpecifier::new_struct(name, members))
        }
    }

    /// Parse struct declaration
    fn parse_struct_declaration(&mut self) -> Result<StructDeclaration> {
        let specifiers = self.parse_specifier_qualifier_list()?;
        let mut declarators = Vec::new();

        if !self.check(TokenKind::Semicolon) {
            loop {
                declarators.push(self.parse_struct_declarator()?);
                if !self.match_token(TokenKind::Comma)? {
                    break;
                }
            }
        }

        self.expect(
            TokenKind::Semicolon,
            "Expected ';' after struct declaration",
        )?;
        Ok(StructDeclaration {
            specifiers,
            declarators,
        })
    }

    /// Parse specifier-qualifier list
    fn parse_specifier_qualifier_list(&mut self) -> Result<Vec<SpecifierQualifier>> {
        let mut list = Vec::new();

        loop {
            match self.peek_kind() {
                Some(TokenKind::Void) => {
                    self.advance()?;
                    list.push(SpecifierQualifier::TypeSpecifier(TypeSpecifier::Void));
                }
                Some(TokenKind::CharKw) => {
                    self.advance()?;
                    list.push(SpecifierQualifier::TypeSpecifier(TypeSpecifier::Char));
                }
                Some(TokenKind::Int) => {
                    self.advance()?;
                    list.push(SpecifierQualifier::TypeSpecifier(TypeSpecifier::Int));
                }
                Some(TokenKind::Const) => {
                    self.advance()?;
                    list.push(SpecifierQualifier::TypeQualifier(TypeQualifier::Const));
                }
                Some(TokenKind::Volatile) => {
                    self.advance()?;
                    list.push(SpecifierQualifier::TypeQualifier(TypeQualifier::Volatile));
                }
                _ => break,
            }
        }

        Ok(list)
    }

    /// Parse struct declarator
    fn parse_struct_declarator(&mut self) -> Result<StructDeclarator> {
        let declarator = if !self.check(TokenKind::Colon) {
            Some(self.parse_declarator()?)
        } else {
            None
        };

        let bit_width = if self.match_token(TokenKind::Colon)? {
            Some(self.parse_expression()?)
        } else {
            None
        };

        Ok(StructDeclarator {
            declarator,
            bit_width,
        })
    }

    /// Parse enum specifier
    fn parse_enum_specifier(&mut self) -> Result<EnumSpecifier> {
        let name = if let Some(TokenKind::Identifier) = self.peek_kind() {
            Some(self.advance()?.lexeme)
        } else {
            None
        };

        let enumerators = if self.check(TokenKind::LBrace) {
            self.advance()?;
            let mut enums = Vec::new();

            loop {
                let token = self.expect(TokenKind::Identifier, "Expected enumerator name")?;
                let enum_name = token.lexeme;

                let value = if self.match_token(TokenKind::Assign)? {
                    Some(self.parse_expression()?)
                } else {
                    None
                };

                enums.push(Enumerator {
                    name: enum_name,
                    value,
                });

                if !self.match_token(TokenKind::Comma)? {
                    break;
                }
                if self.check(TokenKind::RBrace) {
                    break;
                }
            }

            self.expect(TokenKind::RBrace, "Expected '}' after enum")?;
            Some(enums)
        } else {
            None
        };

        Ok(EnumSpecifier { name, enumerators })
    }

    /// Parse declarator
    fn parse_declarator(&mut self) -> Result<Declarator> {
        let pointer = self.parse_pointer()?;
        let direct = self.parse_direct_declarator()?;
        Ok(Declarator::new(pointer, direct))
    }

    /// Parse pointer
    fn parse_pointer(&mut self) -> Result<Option<Pointer>> {
        if !self.check(TokenKind::Star) {
            return Ok(None);
        }

        self.advance()?; // consume *
        let mut qualifiers = Vec::new();

        // Parse type qualifiers
        loop {
            match self.peek_kind() {
                Some(TokenKind::Const) => {
                    self.advance()?;
                    qualifiers.push(TypeQualifier::Const);
                }
                Some(TokenKind::Restrict) => {
                    self.advance()?;
                    qualifiers.push(TypeQualifier::Restrict);
                }
                Some(TokenKind::Volatile) => {
                    self.advance()?;
                    qualifiers.push(TypeQualifier::Volatile);
                }
                _ => break,
            }
        }

        let inner = self.parse_pointer()?;
        Ok(Some(Pointer::new(qualifiers, inner.map(Box::new))))
    }

    /// Parse direct declarator
    fn parse_direct_declarator(&mut self) -> Result<DirectDeclarator> {
        let mut declarator: DirectDeclarator;

        match self.peek_kind() {
            Some(TokenKind::Identifier) => {
                let token = self.advance()?;
                declarator = DirectDeclarator::Identifier(token.lexeme);
            }
            Some(TokenKind::LParen) => {
                self.advance()?;
                let inner = self.parse_declarator()?;
                self.expect(TokenKind::RParen, "Expected ')' after declarator")?;
                declarator = DirectDeclarator::Parenthesized(Box::new(inner));
            }
            _ => {
                let token = self.current()?;
                return Err(Error::syntax(
                    "Expected identifier or '(' in declarator",
                    token.line,
                    token.column,
                ));
            }
        }

        // Parse suffixes (array or function)
        loop {
            match self.peek_kind() {
                Some(TokenKind::LBracket) => {
                    self.advance()?;
                    let size = if self.check(TokenKind::RBracket) {
                        None
                    } else {
                        Some(self.parse_expression()?)
                    };
                    self.expect(TokenKind::RBracket, "Expected ']' after array size")?;
                    declarator = DirectDeclarator::Array(Box::new(declarator), size);
                }
                Some(TokenKind::LParen) => {
                    self.advance()?;
                    let params = if self.check(TokenKind::RParen) {
                        None
                    } else {
                        Some(self.parse_parameter_list()?)
                    };
                    self.expect(TokenKind::RParen, "Expected ')' after parameter list")?;
                    declarator = DirectDeclarator::Function(Box::new(declarator), params);
                }
                _ => break,
            }
        }

        Ok(declarator)
    }

    /// Parse parameter list
    fn parse_parameter_list(&mut self) -> Result<Vec<ParameterDeclaration>> {
        let mut params = Vec::new();

        loop {
            params.push(self.parse_parameter_declaration()?);
            if !self.match_token(TokenKind::Comma)? {
                break;
            }
        }

        Ok(params)
    }

    /// Parse parameter declaration
    fn parse_parameter_declaration(&mut self) -> Result<ParameterDeclaration> {
        let specifiers = self.parse_declaration_specifiers()?;

        // Try to parse declarator or abstract declarator
        if let Some(declarator) = self.try_parse_declarator()? {
            Ok(ParameterDeclaration {
                specifiers,
                declarator: Some(declarator),
                abstract_declarator: None,
            })
        } else if let Some(abstract_decl) = self.try_parse_abstract_declarator()? {
            Ok(ParameterDeclaration {
                specifiers,
                declarator: None,
                abstract_declarator: Some(abstract_decl),
            })
        } else {
            Ok(ParameterDeclaration {
                specifiers,
                declarator: None,
                abstract_declarator: None,
            })
        }
    }

    /// Try to parse a declarator
    fn try_parse_declarator(&mut self) -> Result<Option<Declarator>> {
        // Simple heuristic: if we see * or identifier or (, try to parse
        match self.peek_kind() {
            Some(TokenKind::Star) | Some(TokenKind::Identifier) | Some(TokenKind::LParen) => {
                Ok(Some(self.parse_declarator()?))
            }
            _ => Ok(None),
        }
    }

    /// Try to parse an abstract declarator
    fn try_parse_abstract_declarator(&mut self) -> Result<Option<AbstractDeclarator>> {
        match self.peek_kind() {
            Some(TokenKind::Star) | Some(TokenKind::LParen) | Some(TokenKind::LBracket) => {
                Ok(Some(self.parse_abstract_declarator()?))
            }
            _ => Ok(None),
        }
    }

    /// Parse abstract declarator
    fn parse_abstract_declarator(&mut self) -> Result<AbstractDeclarator> {
        let pointer = self.parse_pointer()?;
        let direct = self.parse_direct_abstract_declarator()?.map(Box::new);
        Ok(AbstractDeclarator { pointer, direct })
    }

    /// Parse direct abstract declarator
    fn parse_direct_abstract_declarator(&mut self) -> Result<Option<DirectAbstractDeclarator>> {
        match self.peek_kind() {
            Some(TokenKind::LParen) => {
                self.advance()?;
                let inner = self.parse_abstract_declarator()?;
                self.expect(TokenKind::RParen, "Expected ')' after abstract declarator")?;
                Ok(Some(DirectAbstractDeclarator::Parenthesized(Box::new(
                    inner,
                ))))
            }
            _ => Ok(None),
        }
    }

    /// Parse declaration
    fn parse_declaration(&mut self) -> Result<Declaration> {
        let specifiers = self.parse_declaration_specifiers()?;
        let mut init_declarators = Vec::new();

        if !self.check(TokenKind::Semicolon) {
            loop {
                let declarator = self.parse_declarator()?;
                let initializer = if self.match_token(TokenKind::Assign)? {
                    Some(self.parse_initializer()?)
                } else {
                    None
                };
                init_declarators.push(InitDeclarator {
                    declarator,
                    initializer,
                });

                if !self.match_token(TokenKind::Comma)? {
                    break;
                }
            }
        }

        self.expect(TokenKind::Semicolon, "Expected ';' after declaration")?;
        Ok(Declaration::new(specifiers, init_declarators))
    }

    /// Parse initializer
    fn parse_initializer(&mut self) -> Result<Initializer> {
        if self.check(TokenKind::LBrace) {
            // Compound initializer - for now, skip the braces and parse inner expression
            self.advance()?;
            // Parse the first expression (simplified)
            let expr = if !self.check(TokenKind::RBrace) {
                self.parse_expression()?
            } else {
                return Err(Error::syntax(
                    "Empty initializer list not supported yet",
                    self.lexer.line(),
                    self.lexer.column(),
                ));
            };
            self.expect(TokenKind::RBrace, "Expected '}' after initializer")?;
            Ok(Initializer::Expression(expr))
        } else {
            Ok(Initializer::Expression(self.parse_expression()?))
        }
    }

    /// Parse compound statement
    fn parse_compound_statement(&mut self) -> Result<CompoundStatement> {
        self.expect(
            TokenKind::LBrace,
            "Expected '{' to start compound statement",
        )?;

        let mut items = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            items.push(self.parse_block_item()?);
        }

        self.expect(TokenKind::RBrace, "Expected '}' to end compound statement")?;
        Ok(CompoundStatement::new(items))
    }

    /// Parse block item
    fn parse_block_item(&mut self) -> Result<BlockItem> {
        // Check if this is a declaration (starts with type specifier)
        match self.peek_kind() {
            Some(TokenKind::Int)
            | Some(TokenKind::CharKw)
            | Some(TokenKind::Void)
            | Some(TokenKind::FloatKw)
            | Some(TokenKind::Double)
            | Some(TokenKind::Struct)
            | Some(TokenKind::Union)
            | Some(TokenKind::Enum)
            | Some(TokenKind::Typedef)
            | Some(TokenKind::Extern)
            | Some(TokenKind::Static)
            | Some(TokenKind::Auto)
            | Some(TokenKind::Register)
            | Some(TokenKind::Const)
            | Some(TokenKind::Volatile)
            | Some(TokenKind::Inline)
            | Some(TokenKind::Long)
            | Some(TokenKind::Short)
            | Some(TokenKind::Signed)
            | Some(TokenKind::Unsigned) => Ok(BlockItem::Declaration(self.parse_declaration()?)),
            _ => Ok(BlockItem::Statement(self.parse_statement()?)),
        }
    }

    /// Parse statement
    fn parse_statement(&mut self) -> Result<Statement> {
        match self.peek_kind() {
            Some(TokenKind::Identifier) => {
                // Check if it's a label
                // This is simplified; in reality, we need to look ahead
                self.parse_expression_statement()
            }
            Some(TokenKind::Case) | Some(TokenKind::Default) => self.parse_labeled_statement(),
            Some(TokenKind::LBrace) => Ok(Statement::Compound(self.parse_compound_statement()?)),
            Some(TokenKind::If) | Some(TokenKind::Switch) => self.parse_selection_statement(),
            Some(TokenKind::While) | Some(TokenKind::Do) | Some(TokenKind::For) => {
                self.parse_iteration_statement()
            }
            Some(TokenKind::Goto)
            | Some(TokenKind::Continue)
            | Some(TokenKind::Break)
            | Some(TokenKind::Return) => self.parse_jump_statement(),
            _ => self.parse_expression_statement(),
        }
    }

    /// Parse labeled statement
    fn parse_labeled_statement(&mut self) -> Result<Statement> {
        match self.peek_kind() {
            Some(TokenKind::Case) => {
                self.advance()?;
                let expr = self.parse_expression()?;
                self.expect(TokenKind::Colon, "Expected ':' after case")?;
                let stmt = self.parse_statement()?;
                Ok(Statement::Labeled(LabeledStatement::Case(
                    expr,
                    Box::new(stmt),
                )))
            }
            Some(TokenKind::Default) => {
                self.advance()?;
                self.expect(TokenKind::Colon, "Expected ':' after default")?;
                let stmt = self.parse_statement()?;
                Ok(Statement::Labeled(LabeledStatement::Default(Box::new(
                    stmt,
                ))))
            }
            _ => {
                let token = self.expect(TokenKind::Identifier, "Expected label name")?;
                let label = token.lexeme;
                self.expect(TokenKind::Colon, "Expected ':' after label")?;
                let stmt = self.parse_statement()?;
                Ok(Statement::Labeled(LabeledStatement::Label(
                    label,
                    Box::new(stmt),
                )))
            }
        }
    }

    /// Parse expression statement
    fn parse_expression_statement(&mut self) -> Result<Statement> {
        let expr = if self.check(TokenKind::Semicolon) {
            None
        } else {
            Some(self.parse_expression()?)
        };
        self.expect(TokenKind::Semicolon, "Expected ';' after expression")?;
        Ok(Statement::Expression(ExpressionStatement {
            expression: expr,
        }))
    }

    /// Parse selection statement
    fn parse_selection_statement(&mut self) -> Result<Statement> {
        match self.peek_kind() {
            Some(TokenKind::If) => {
                self.advance()?;
                self.expect(TokenKind::LParen, "Expected '(' after 'if'")?;
                let condition = self.parse_expression()?;
                self.expect(TokenKind::RParen, "Expected ')' after if condition")?;
                let then_branch = self.parse_statement()?;
                let else_branch = if self.match_token(TokenKind::Else)? {
                    Some(Box::new(self.parse_statement()?))
                } else {
                    None
                };
                Ok(Statement::Selection(SelectionStatement::If(
                    condition,
                    Box::new(then_branch),
                    else_branch,
                )))
            }
            Some(TokenKind::Switch) => {
                self.advance()?;
                self.expect(TokenKind::LParen, "Expected '(' after 'switch'")?;
                let expr = self.parse_expression()?;
                self.expect(TokenKind::RParen, "Expected ')' after switch expression")?;
                let body = self.parse_statement()?;
                Ok(Statement::Selection(SelectionStatement::Switch(
                    expr,
                    Box::new(body),
                )))
            }
            _ => Err(Error::syntax("Expected selection statement", 0, 0)),
        }
    }

    /// Parse iteration statement
    fn parse_iteration_statement(&mut self) -> Result<Statement> {
        match self.peek_kind() {
            Some(TokenKind::While) => {
                self.advance()?;
                self.expect(TokenKind::LParen, "Expected '(' after 'while'")?;
                let condition = self.parse_expression()?;
                self.expect(TokenKind::RParen, "Expected ')' after while condition")?;
                let body = self.parse_statement()?;
                Ok(Statement::Iteration(IterationStatement::While(
                    condition,
                    Box::new(body),
                )))
            }
            Some(TokenKind::Do) => {
                self.advance()?;
                let body = self.parse_statement()?;
                self.expect(TokenKind::While, "Expected 'while' after 'do'")?;
                self.expect(TokenKind::LParen, "Expected '(' after 'while'")?;
                let condition = self.parse_expression()?;
                self.expect(TokenKind::RParen, "Expected ')' after do-while condition")?;
                self.expect(TokenKind::Semicolon, "Expected ';' after do-while")?;
                Ok(Statement::Iteration(IterationStatement::DoWhile(
                    Box::new(body),
                    condition,
                )))
            }
            Some(TokenKind::For) => {
                self.advance()?;
                self.expect(TokenKind::LParen, "Expected '(' after 'for'")?;

                let init = if self.check(TokenKind::Semicolon) {
                    ForInit::Expression(None)
                } else if self.is_type_specifier_start() {
                    ForInit::Declaration(self.parse_declaration()?)
                } else {
                    ForInit::Expression(Some(self.parse_expression()?))
                };

                if matches!(init, ForInit::Expression(_)) {
                    self.expect(TokenKind::Semicolon, "Expected ';' after for init")?;
                }

                let condition = if self.check(TokenKind::Semicolon) {
                    None
                } else {
                    Some(self.parse_expression()?)
                };
                self.expect(TokenKind::Semicolon, "Expected ';' after for condition")?;

                let update = if self.check(TokenKind::RParen) {
                    None
                } else {
                    Some(self.parse_expression()?)
                };
                self.expect(TokenKind::RParen, "Expected ')' after for clauses")?;

                let body = self.parse_statement()?;
                Ok(Statement::Iteration(IterationStatement::For(
                    init,
                    condition,
                    update,
                    Box::new(body),
                )))
            }
            _ => Err(Error::syntax("Expected iteration statement", 0, 0)),
        }
    }

    /// Check if the current position starts a type specifier
    fn is_type_specifier_start(&self) -> bool {
        matches!(
            self.peek_kind(),
            Some(TokenKind::Int)
                | Some(TokenKind::CharKw)
                | Some(TokenKind::Void)
                | Some(TokenKind::FloatKw)
                | Some(TokenKind::Double)
                | Some(TokenKind::Struct)
                | Some(TokenKind::Union)
                | Some(TokenKind::Enum)
                | Some(TokenKind::Typedef)
                | Some(TokenKind::Extern)
                | Some(TokenKind::Static)
                | Some(TokenKind::Long)
                | Some(TokenKind::Short)
                | Some(TokenKind::Signed)
                | Some(TokenKind::Unsigned)
        )
    }

    /// Parse jump statement
    fn parse_jump_statement(&mut self) -> Result<Statement> {
        match self.peek_kind() {
            Some(TokenKind::Goto) => {
                self.advance()?;
                let token = self.expect(TokenKind::Identifier, "Expected label after 'goto'")?;
                let label = token.lexeme;
                self.expect(TokenKind::Semicolon, "Expected ';' after goto")?;
                Ok(Statement::Jump(JumpStatement::Goto(label)))
            }
            Some(TokenKind::Continue) => {
                self.advance()?;
                self.expect(TokenKind::Semicolon, "Expected ';' after 'continue'")?;
                Ok(Statement::Jump(JumpStatement::Continue))
            }
            Some(TokenKind::Break) => {
                self.advance()?;
                self.expect(TokenKind::Semicolon, "Expected ';' after 'break'")?;
                Ok(Statement::Jump(JumpStatement::Break))
            }
            Some(TokenKind::Return) => {
                self.advance()?;
                let value = if self.check(TokenKind::Semicolon) {
                    None
                } else {
                    Some(self.parse_expression()?)
                };
                self.expect(TokenKind::Semicolon, "Expected ';' after return")?;
                Ok(Statement::Jump(JumpStatement::Return(value)))
            }
            _ => Err(Error::syntax("Expected jump statement", 0, 0)),
        }
    }

    /// Parse expression (entry point for expression parsing)
    fn parse_expression(&mut self) -> Result<Expression> {
        self.parse_assignment_expression()
    }

    /// Parse assignment expression
    fn parse_assignment_expression(&mut self) -> Result<Expression> {
        let lhs = self.parse_conditional_expression()?;

        match self.peek_kind() {
            Some(TokenKind::Assign) => {
                self.advance()?;
                let rhs = self.parse_assignment_expression()?;
                Ok(Expression::Assign(Box::new(lhs), Box::new(rhs)))
            }
            Some(TokenKind::PlusAssign) => {
                self.advance()?;
                let rhs = self.parse_assignment_expression()?;
                Ok(Expression::AddAssign(Box::new(lhs), Box::new(rhs)))
            }
            Some(TokenKind::MinusAssign) => {
                self.advance()?;
                let rhs = self.parse_assignment_expression()?;
                Ok(Expression::SubtractAssign(Box::new(lhs), Box::new(rhs)))
            }
            Some(TokenKind::MulAssign) => {
                self.advance()?;
                let rhs = self.parse_assignment_expression()?;
                Ok(Expression::MultiplyAssign(Box::new(lhs), Box::new(rhs)))
            }
            Some(TokenKind::DivAssign) => {
                self.advance()?;
                let rhs = self.parse_assignment_expression()?;
                Ok(Expression::DivideAssign(Box::new(lhs), Box::new(rhs)))
            }
            Some(TokenKind::ModAssign) => {
                self.advance()?;
                let rhs = self.parse_assignment_expression()?;
                Ok(Expression::ModuloAssign(Box::new(lhs), Box::new(rhs)))
            }
            Some(TokenKind::ShiftLeft) => {
                // Check if it's <<= or just <<
                if self.peek_char2() == Some('=') {
                    self.advance()?;
                    self.advance()?;
                    let rhs = self.parse_assignment_expression()?;
                    Ok(Expression::ShiftLeftAssign(Box::new(lhs), Box::new(rhs)))
                } else {
                    Ok(lhs)
                }
            }
            Some(TokenKind::ShiftRight) => {
                // Check if it's >>= or just >>
                if self.peek_char2() == Some('=') {
                    self.advance()?;
                    self.advance()?;
                    let rhs = self.parse_assignment_expression()?;
                    Ok(Expression::ShiftRightAssign(Box::new(lhs), Box::new(rhs)))
                } else {
                    Ok(lhs)
                }
            }
            _ => Ok(lhs),
        }
    }

    /// Peek at the second next character from lexer
    fn peek_char2(&self) -> Option<char> {
        // Simplified - in production, you'd properly implement this
        None
    }

    /// Parse conditional expression
    fn parse_conditional_expression(&mut self) -> Result<Expression> {
        let condition = self.parse_logical_or_expression()?;

        if self.match_token(TokenKind::Question)? {
            let true_branch = self.parse_expression()?;
            self.expect(TokenKind::Colon, "Expected ':' in conditional expression")?;
            let false_branch = self.parse_conditional_expression()?;
            Ok(Expression::Conditional(
                Box::new(condition),
                Box::new(true_branch),
                Box::new(false_branch),
            ))
        } else {
            Ok(condition)
        }
    }

    /// Parse logical OR expression
    fn parse_logical_or_expression(&mut self) -> Result<Expression> {
        let mut left = self.parse_logical_and_expression()?;

        while self.match_token(TokenKind::LogicalOr)? {
            let right = self.parse_logical_and_expression()?;
            left = Expression::LogicalOr(Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    /// Parse logical AND expression
    fn parse_logical_and_expression(&mut self) -> Result<Expression> {
        let mut left = self.parse_bitwise_or_expression()?;

        while self.match_token(TokenKind::LogicalAnd)? {
            let right = self.parse_bitwise_or_expression()?;
            left = Expression::LogicalAnd(Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    /// Parse bitwise OR expression
    fn parse_bitwise_or_expression(&mut self) -> Result<Expression> {
        let mut left = self.parse_bitwise_xor_expression()?;

        while self.check(TokenKind::Or) {
            self.advance()?;
            let right = self.parse_bitwise_xor_expression()?;
            left = Expression::BitwiseOr(Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    /// Parse bitwise XOR expression
    fn parse_bitwise_xor_expression(&mut self) -> Result<Expression> {
        let mut left = self.parse_bitwise_and_expression()?;

        while self.check(TokenKind::Xor) {
            self.advance()?;
            let right = self.parse_bitwise_and_expression()?;
            left = Expression::BitwiseXor(Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    /// Parse bitwise AND expression
    fn parse_bitwise_and_expression(&mut self) -> Result<Expression> {
        let mut left = self.parse_equality_expression()?;

        while self.check(TokenKind::And) {
            self.advance()?;
            let right = self.parse_equality_expression()?;
            left = Expression::BitwiseAnd(Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    /// Parse equality expression
    fn parse_equality_expression(&mut self) -> Result<Expression> {
        let mut left = self.parse_relational_expression()?;

        loop {
            if self.match_token(TokenKind::Eq)? {
                let right = self.parse_relational_expression()?;
                left = Expression::Equal(Box::new(left), Box::new(right));
            } else if self.match_token(TokenKind::Ne)? {
                let right = self.parse_relational_expression()?;
                left = Expression::NotEqual(Box::new(left), Box::new(right));
            } else {
                break;
            }
        }

        Ok(left)
    }

    /// Parse relational expression
    fn parse_relational_expression(&mut self) -> Result<Expression> {
        let mut left = self.parse_shift_expression()?;

        loop {
            if self.match_token(TokenKind::Lt)? {
                let right = self.parse_shift_expression()?;
                left = Expression::LessThan(Box::new(left), Box::new(right));
            } else if self.match_token(TokenKind::Le)? {
                let right = self.parse_shift_expression()?;
                left = Expression::LessThanOrEqual(Box::new(left), Box::new(right));
            } else if self.match_token(TokenKind::Gt)? {
                let right = self.parse_shift_expression()?;
                left = Expression::GreaterThan(Box::new(left), Box::new(right));
            } else if self.match_token(TokenKind::Ge)? {
                let right = self.parse_shift_expression()?;
                left = Expression::GreaterThanOrEqual(Box::new(left), Box::new(right));
            } else {
                break;
            }
        }

        Ok(left)
    }

    /// Parse shift expression
    fn parse_shift_expression(&mut self) -> Result<Expression> {
        let mut left = self.parse_additive_expression()?;

        loop {
            if self.match_token(TokenKind::ShiftLeft)? {
                let right = self.parse_additive_expression()?;
                left = Expression::ShiftLeft(Box::new(left), Box::new(right));
            } else if self.match_token(TokenKind::ShiftRight)? {
                let right = self.parse_additive_expression()?;
                left = Expression::ShiftRight(Box::new(left), Box::new(right));
            } else {
                break;
            }
        }

        Ok(left)
    }

    /// Parse additive expression
    fn parse_additive_expression(&mut self) -> Result<Expression> {
        let mut left = self.parse_multiplicative_expression()?;

        loop {
            if self.match_token(TokenKind::Plus)? {
                let right = self.parse_multiplicative_expression()?;
                left = Expression::Add(Box::new(left), Box::new(right));
            } else if self.match_token(TokenKind::Minus)? {
                let right = self.parse_multiplicative_expression()?;
                left = Expression::Subtract(Box::new(left), Box::new(right));
            } else {
                break;
            }
        }

        Ok(left)
    }

    /// Parse multiplicative expression
    fn parse_multiplicative_expression(&mut self) -> Result<Expression> {
        let mut left = self.parse_cast_expression()?;

        loop {
            if self.match_token(TokenKind::Star)? {
                let right = self.parse_cast_expression()?;
                left = Expression::Multiply(Box::new(left), Box::new(right));
            } else if self.match_token(TokenKind::Slash)? {
                let right = self.parse_cast_expression()?;
                left = Expression::Divide(Box::new(left), Box::new(right));
            } else if self.match_token(TokenKind::Percent)? {
                let right = self.parse_cast_expression()?;
                left = Expression::Modulo(Box::new(left), Box::new(right));
            } else {
                break;
            }
        }

        Ok(left)
    }

    /// Parse cast expression
    fn parse_cast_expression(&mut self) -> Result<Expression> {
        if self.check(TokenKind::LParen) {
            // Check if it's a cast or parenthesized expression
            // This is a simplified check
            let checkpoint = self.save_checkpoint();
            self.advance()?;

            // Try to parse as type name
            if self.is_type_specifier_start() {
                // This might be a cast
                match self.parse_type_name() {
                    Ok(type_name) => {
                        if self.check(TokenKind::RParen) {
                            self.advance()?;
                            let expr = self.parse_cast_expression()?;
                            return Ok(Expression::Cast(type_name, Box::new(expr)));
                        }
                    }
                    Err(_) => {}
                }
            }

            // Restore and try as parenthesized expression
            self.restore_checkpoint(checkpoint);
        }

        self.parse_unary_expression()
    }

    /// Parse type name
    fn parse_type_name(&mut self) -> Result<TypeName> {
        let specifiers = self.parse_specifier_qualifier_list()?;
        let abstract_declarator = self.try_parse_abstract_declarator()?;
        Ok(TypeName {
            specifiers,
            abstract_declarator,
        })
    }

    /// Parse unary expression
    fn parse_unary_expression(&mut self) -> Result<Expression> {
        match self.peek_kind() {
            Some(TokenKind::Increment) => {
                self.advance()?;
                let expr = self.parse_unary_expression()?;
                Ok(Expression::PreIncrement(Box::new(expr)))
            }
            Some(TokenKind::Decrement) => {
                self.advance()?;
                let expr = self.parse_unary_expression()?;
                Ok(Expression::PreDecrement(Box::new(expr)))
            }
            Some(TokenKind::And) => {
                self.advance()?;
                let expr = self.parse_cast_expression()?;
                Ok(Expression::AddressOf(Box::new(expr)))
            }
            Some(TokenKind::Star) => {
                self.advance()?;
                let expr = self.parse_cast_expression()?;
                Ok(Expression::Dereference(Box::new(expr)))
            }
            Some(TokenKind::Plus) => {
                self.advance()?;
                let expr = self.parse_cast_expression()?;
                Ok(Expression::UnaryPlus(Box::new(expr)))
            }
            Some(TokenKind::Minus) => {
                self.advance()?;
                let expr = self.parse_cast_expression()?;
                Ok(Expression::UnaryMinus(Box::new(expr)))
            }
            Some(TokenKind::Tilde) => {
                self.advance()?;
                let expr = self.parse_cast_expression()?;
                Ok(Expression::BitwiseNot(Box::new(expr)))
            }
            Some(TokenKind::Not) => {
                self.advance()?;
                let expr = self.parse_cast_expression()?;
                Ok(Expression::LogicalNot(Box::new(expr)))
            }
            Some(TokenKind::Sizeof) => {
                self.advance()?;
                if self.check(TokenKind::LParen) && self.is_type_specifier_start_at(1) {
                    self.advance()?;
                    let type_name = self.parse_type_name()?;
                    self.expect(TokenKind::RParen, "Expected ')' after sizeof type")?;
                    Ok(Expression::SizeofType(
                        type_name.specifiers,
                        type_name.abstract_declarator.map(Box::new),
                    ))
                } else {
                    let expr = self.parse_unary_expression()?;
                    Ok(Expression::SizeofExpression(Box::new(expr)))
                }
            }
            _ => self.parse_postfix_expression(),
        }
    }

    /// Check if token at offset is a type specifier start
    fn is_type_specifier_start_at(&self, _offset: usize) -> bool {
        // Simplified - in production, properly implement lookahead
        true
    }

    /// Parse postfix expression
    fn parse_postfix_expression(&mut self) -> Result<Expression> {
        let mut expr = self.parse_primary_expression()?;

        loop {
            match self.peek_kind() {
                Some(TokenKind::LBracket) => {
                    self.advance()?;
                    let index = self.parse_expression()?;
                    self.expect(TokenKind::RBracket, "Expected ']' after array index")?;
                    expr = Expression::ArrayAccess(Box::new(expr), Box::new(index));
                }
                Some(TokenKind::LParen) => {
                    self.advance()?;
                    let args = if self.check(TokenKind::RParen) {
                        Vec::new()
                    } else {
                        self.parse_argument_expression_list()?
                    };
                    self.expect(TokenKind::RParen, "Expected ')' after function arguments")?;
                    expr = Expression::FunctionCall(Box::new(expr), args);
                }
                Some(TokenKind::Dot) => {
                    self.advance()?;
                    let token =
                        self.expect(TokenKind::Identifier, "Expected member name after '.'")?;
                    expr = Expression::MemberAccess(Box::new(expr), token.lexeme);
                }
                Some(TokenKind::Arrow) => {
                    self.advance()?;
                    let token =
                        self.expect(TokenKind::Identifier, "Expected member name after '->'")?;
                    expr = Expression::PointerMemberAccess(Box::new(expr), token.lexeme);
                }
                Some(TokenKind::Increment) => {
                    self.advance()?;
                    expr = Expression::PostIncrement(Box::new(expr));
                }
                Some(TokenKind::Decrement) => {
                    self.advance()?;
                    expr = Expression::PostDecrement(Box::new(expr));
                }
                _ => break,
            }
        }

        Ok(expr)
    }

    /// Parse argument expression list
    fn parse_argument_expression_list(&mut self) -> Result<Vec<Expression>> {
        let mut args = Vec::new();

        loop {
            args.push(self.parse_assignment_expression()?);
            if !self.match_token(TokenKind::Comma)? {
                break;
            }
        }

        Ok(args)
    }

    /// Parse primary expression
    fn parse_primary_expression(&mut self) -> Result<Expression> {
        match self.peek_kind() {
            Some(TokenKind::Identifier) => {
                let token = self.advance()?;
                Ok(Expression::Identifier(token.lexeme))
            }
            Some(TokenKind::Integer(val)) => {
                self.advance()?;
                Ok(Expression::Integer(val))
            }
            Some(TokenKind::FloatLit(val)) => {
                self.advance()?;
                Ok(Expression::Float(val))
            }
            Some(TokenKind::CharLit(c)) => {
                self.advance()?;
                Ok(Expression::Char(c))
            }
            Some(TokenKind::StringLit) => {
                let token = self.advance()?;
                // Remove quotes from string literal
                let s = token.lexeme;
                let content = &s[1..s.len() - 1];
                Ok(Expression::String(content.to_string()))
            }
            Some(TokenKind::LParen) => {
                self.advance()?;
                let expr = self.parse_expression()?;
                self.expect(TokenKind::RParen, "Expected ')' after expression")?;
                Ok(Expression::Parenthesized(Box::new(expr)))
            }
            _ => {
                let token = self.current()?;
                Err(Error::syntax(
                    format!("Unexpected token: {:?}", token.kind),
                    token.line,
                    token.column,
                ))
            }
        }
    }
}

/// Checkpoint for parser backtracking
#[derive(Clone)]
struct ParserCheckpoint {
    current: Option<Token>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_function() {
        let source = "int main(void) { return 0; }";
        let lexer = Lexer::new(source);
        let mut parser = Parser::new(lexer);

        let result = parser.parse_translation_unit();
        assert!(result.is_ok());

        let ast = result.unwrap();
        assert_eq!(ast.declarations.len(), 1);
    }

    #[test]
    fn test_parse_variable_declaration() {
        let source = "int x = 42;";
        let lexer = Lexer::new(source);
        let mut parser = Parser::new(lexer);

        let result = parser.parse_translation_unit();
        if let Err(ref e) = result {
            eprintln!("Parse error: {}", e);
        }
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_if_statement() {
        let source = r#"
            int test(int x) {
                if (x > 0) {
                    return 1;
                } else {
                    return 0;
                }
            }
        "#;
        let lexer = Lexer::new(source);
        let mut parser = Parser::new(lexer);

        let result = parser.parse_translation_unit();
        assert!(result.is_ok());
    }
}
