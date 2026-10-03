//! Read-only expression traversal used for frontend preparation.
use crate::ast::*;

pub fn expression(expr: &Expression, f: &mut impl FnMut(&Expression)) {
    use Expression::*;
    f(expr);
    match expr {
        Parenthesized(x)
        | PostIncrement(x)
        | PostDecrement(x)
        | PreIncrement(x)
        | PreDecrement(x)
        | AddressOf(x)
        | Dereference(x)
        | UnaryPlus(x)
        | UnaryMinus(x)
        | BitwiseNot(x)
        | LogicalNot(x)
        | SizeofExpression(x)
        | Cast(_, x)
        | MemberAccess(x, _)
        | PointerMemberAccess(x, _) => expression(x, f),
        ArrayAccess(a, b)
        | Multiply(a, b)
        | Divide(a, b)
        | Modulo(a, b)
        | Add(a, b)
        | Subtract(a, b)
        | ShiftLeft(a, b)
        | ShiftRight(a, b)
        | LessThan(a, b)
        | LessThanOrEqual(a, b)
        | GreaterThan(a, b)
        | GreaterThanOrEqual(a, b)
        | Equal(a, b)
        | NotEqual(a, b)
        | BitwiseAnd(a, b)
        | BitwiseXor(a, b)
        | BitwiseOr(a, b)
        | LogicalAnd(a, b)
        | LogicalOr(a, b)
        | Assign(a, b)
        | MultiplyAssign(a, b)
        | DivideAssign(a, b)
        | ModuloAssign(a, b)
        | AddAssign(a, b)
        | SubtractAssign(a, b)
        | ShiftLeftAssign(a, b)
        | ShiftRightAssign(a, b)
        | BitwiseAndAssign(a, b)
        | BitwiseXorAssign(a, b)
        | BitwiseOrAssign(a, b)
        | Comma(a, b) => {
            expression(a, f);
            expression(b, f);
        }
        Conditional(a, b, c) => {
            expression(a, f);
            expression(b, f);
            expression(c, f);
        }
        FunctionCall(c, args) => {
            expression(c, f);
            for arg in args {
                expression(arg, f);
            }
        }
        _ => {}
    }
}
pub fn initializer(init: &Initializer, f: &mut impl FnMut(&Expression)) {
    match init {
        Initializer::Expression(expr) => expression(expr, f),
        Initializer::List(items) => {
            for item in items {
                initializer(item, f);
            }
        }
    }
}
pub fn declaration(decl: &Declaration, f: &mut impl FnMut(&Expression)) {
    for item in &decl.init_declarators {
        if let Some(init) = &item.initializer {
            initializer(init, f);
        }
    }
}
pub fn compound(body: &CompoundStatement, f: &mut impl FnMut(&Expression)) {
    for item in &body.items {
        match item {
            BlockItem::Declaration(d) => declaration(d, f),
            BlockItem::Statement(s) => statement(s, f),
        }
    }
}
pub fn statement(stmt: &Statement, f: &mut impl FnMut(&Expression)) {
    match stmt {
        Statement::Compound(body) => compound(body, f),
        Statement::Expression(s) => {
            if let Some(e) = &s.expression {
                expression(e, f);
            }
        }
        Statement::Selection(SelectionStatement::If(c, a, b)) => {
            expression(c, f);
            statement(a, f);
            if let Some(b) = b {
                statement(b, f);
            }
        }
        Statement::Selection(SelectionStatement::Switch(e, s)) => {
            expression(e, f);
            statement(s, f);
        }
        Statement::Iteration(IterationStatement::While(e, s))
        | Statement::Iteration(IterationStatement::DoWhile(s, e)) => {
            expression(e, f);
            statement(s, f);
        }
        Statement::Iteration(IterationStatement::For(init, c, step, body)) => {
            match init {
                ForInit::Declaration(d) => declaration(d, f),
                ForInit::Expression(e) => {
                    if let Some(e) = e {
                        expression(e, f);
                    }
                }
            }
            if let Some(e) = c {
                expression(e, f);
            }
            if let Some(e) = step {
                expression(e, f);
            }
            statement(body, f);
        }
        Statement::Labeled(LabeledStatement::Case(e, s)) => {
            expression(e, f);
            statement(s, f);
        }
        Statement::Labeled(LabeledStatement::Default(s))
        | Statement::Labeled(LabeledStatement::Label(_, s)) => statement(s, f),
        Statement::Jump(JumpStatement::Return(Some(e))) => expression(e, f),
        _ => {}
    }
}
