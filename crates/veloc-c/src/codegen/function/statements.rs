use super::*;

impl Function<'_, '_> {
    fn jump(&mut self, block: Block) {
        if !self.builder.is_current_block_terminated() {
            self.builder.ins().jump(block, &[]);
        }
    }
    /// Conditions have control-flow semantics. Keep short-circuit operators as
    /// edges instead of materializing an integer and immediately testing it.
    fn branch(&mut self, expr: &Expression, yes: Block, no: Block) -> Result<()> {
        match expr {
            Expression::Parenthesized(inner) => self.branch(inner, yes, no),
            Expression::LogicalNot(inner) => self.branch(inner, no, yes),
            Expression::LogicalAnd(a, b) => {
                let rhs = self.builder.create_block();
                self.branch(a, rhs, no)?;
                self.builder.switch_to_block(rhs);
                self.branch(b, yes, no)
            }
            Expression::LogicalOr(a, b) => {
                let rhs = self.builder.create_block();
                self.branch(a, yes, rhs)?;
                self.builder.switch_to_block(rhs);
                self.branch(b, yes, no)
            }
            _ => {
                let value = self.expr(expr)?;
                let condition = self.truth(value)?;
                self.builder.ins().br(condition, yes, &[], no, &[]);
                Ok(())
            }
        }
    }
    pub(super) fn compound(&mut self, body: &CompoundStatement) -> Result<()> {
        self.scopes.push(HashMap::new());
        for item in &body.items {
            match item {
                BlockItem::Declaration(d) if !self.builder.is_current_block_terminated() => {
                    self.declaration(d)?
                }
                BlockItem::Statement(s) => self.statement(s)?,
                _ => {}
            }
        }
        self.scopes.pop();
        Ok(())
    }
    fn declaration(&mut self, decl: &Declaration) -> Result<()> {
        let base = self.types.specifiers(&decl.specifiers)?;
        if decl
            .specifiers
            .contains(&DeclarationSpecifier::StorageClass(
                StorageClassSpecifier::Static,
            ))
        {
            return fail("block-scope static objects are not supported yet");
        }
        for item in &decl.init_declarators {
            let name = item.declarator.name();
            let mut ty = self.types.declarator(base.clone(), &item.declarator)?;
            if decl
                .specifiers
                .contains(&DeclarationSpecifier::StorageClass(
                    StorageClassSpecifier::Typedef,
                ))
            {
                self.types.typedefs.insert(name.into(), ty);
                continue;
            }
            complete_array(&mut ty, item.initializer.as_ref())?;
            let place = self.local(name, ty)?;
            if let Some(init) = &item.initializer {
                self.initialize(&place, init)?;
            }
        }
        Ok(())
    }
    fn initialize(&mut self, place: &Place, init: &Initializer) -> Result<()> {
        match (place.ty.plain(), init) {
            (CType::Array(element, count), Initializer::List(items)) => {
                if items.len() > *count {
                    return fail("too many array initializers");
                }
                let Location::Memory(ptr) = place.location else {
                    return fail("array storage missing");
                };
                let size = self.types.layout(element)?.0;
                for index in 0..*count {
                    let at = self.builder.ins().ptr_offset(
                        ptr,
                        (index * size)
                            .try_into()
                            .map_err(|_| Error::semantic("array too large", 0, 0))?,
                    );
                    let target = Place {
                        location: Location::Memory(at),
                        ty: *element.clone(),
                    };
                    let zero = Initializer::Expression(Expression::Integer(IntegerLiteral::int(0)));
                    self.initialize(&target, items.get(index).unwrap_or(&zero))?;
                }
            }
            (CType::Array(element, count), Initializer::Expression(Expression::String(text))) => {
                if !matches!(element.plain(), CType::Int(8, _)) {
                    return fail("string requires character array");
                }
                let Location::Memory(ptr) = place.location else {
                    return fail("array storage missing");
                };
                let bytes = string_bytes(text)?;
                for index in 0..*count {
                    let v = self.integer(
                        bytes.get(index).copied().unwrap_or(0) as u64,
                        *element.clone(),
                    )?;
                    self.builder
                        .ins()
                        .store(ptr, v.value, index as u32, Self::flags(element));
                }
            }
            (CType::Record(id), Initializer::List(items)) => {
                let record = &self.types.records[*id];
                let fields: Vec<_> = record
                    .members
                    .iter()
                    .take(if record.is_union {
                        1
                    } else {
                        record.members.len()
                    })
                    .cloned()
                    .collect();
                if items.len() > fields.len() {
                    return fail("too many record initializers");
                }
                let Location::Memory(ptr) = place.location else {
                    return fail("record storage missing");
                };
                for (index, field) in fields.iter().enumerate() {
                    let at = self.builder.ins().ptr_offset(ptr, field.offset as i32);
                    let target = Place {
                        location: Location::Memory(at),
                        ty: field.ty.clone(),
                    };
                    let zero = Initializer::Expression(Expression::Integer(IntegerLiteral::int(0)));
                    self.initialize(&target, items.get(index).unwrap_or(&zero))?;
                }
            }
            (_, Initializer::List(items)) if items.len() == 1 => {
                self.initialize(place, &items[0])?
            }
            (_, Initializer::Expression(e)) => {
                let v = self.expr(e)?;
                self.write(place, v)?;
            }
            _ => return fail("unsupported local initializer"),
        }
        Ok(())
    }
    fn statement(&mut self, stmt: &Statement) -> Result<()> {
        if self.builder.is_current_block_terminated()
            && !matches!(stmt, Statement::Compound(_) | Statement::Labeled(_))
        {
            return Ok(());
        }
        match stmt {
            Statement::Compound(body) => self.compound(body)?,
            Statement::Expression(s) => {
                if let Some(e) = &s.expression {
                    self.expr(e)?;
                }
            }
            Statement::Jump(JumpStatement::Return(e)) => {
                if let Some(e) = e {
                    let v = self.expr(e)?;
                    if self.signature.result == CType::Void {
                        return fail("value returned from void function");
                    }
                    let v = self.cast(v, &self.signature.result)?;
                    let v = self.cast(v, &self.signature.result.abi_type())?;
                    self.builder.ins().ret(&[v.value]);
                } else {
                    if self.signature.result != CType::Void {
                        return fail("missing return value");
                    }
                    self.builder.ins().ret(&[]);
                }
            }
            Statement::Jump(JumpStatement::Break) => {
                let block = *self
                    .breaks
                    .last()
                    .ok_or_else(|| Error::semantic("break outside loop/switch", 0, 0))?;
                self.jump(block);
            }
            Statement::Jump(JumpStatement::Continue) => {
                let block = *self
                    .continues
                    .last()
                    .ok_or_else(|| Error::semantic("continue outside loop", 0, 0))?;
                self.jump(block);
            }
            Statement::Selection(SelectionStatement::If(cond, yes, no)) => {
                let then_block = self.builder.create_block();
                let else_block = self.builder.create_block();
                let merge = self.builder.create_block();
                self.branch(cond, then_block, else_block)?;
                self.builder.switch_to_block(then_block);
                self.statement(yes)?;
                self.jump(merge);
                self.builder.switch_to_block(else_block);
                if let Some(no) = no {
                    self.statement(no)?;
                }
                self.jump(merge);
                self.builder.switch_to_block(merge);
                if self.builder.func().cfg().preds(merge).is_empty() {
                    self.builder.ins().unreachable();
                }
            }
            Statement::Iteration(IterationStatement::While(cond, body)) => {
                self.loop_body(None, Some(cond), None, body, false)?
            }
            Statement::Iteration(IterationStatement::DoWhile(body, cond)) => {
                self.loop_body(None, Some(cond), None, body, true)?
            }
            Statement::Iteration(IterationStatement::For(init, cond, step, body)) => {
                self.loop_body(Some(init), cond.as_ref(), step.as_ref(), body, false)?
            }
            Statement::Selection(SelectionStatement::Switch(expr, body)) => {
                self.switch(expr, body)?
            }
            Statement::Labeled(LabeledStatement::Case(_, body))
            | Statement::Labeled(LabeledStatement::Default(body)) => {
                let block = self
                    .cases
                    .last_mut()
                    .and_then(|q| q.pop_front())
                    .ok_or_else(|| Error::semantic("case outside switch", 0, 0))?;
                self.jump(block);
                self.builder.switch_to_block(block);
                self.statement(body)?;
            }
            _ => return fail("goto and labels are not supported yet"),
        }
        Ok(())
    }
    fn loop_body(
        &mut self,
        init: Option<&ForInit>,
        cond: Option<&Expression>,
        step: Option<&Expression>,
        body: &Statement,
        do_first: bool,
    ) -> Result<()> {
        self.scopes.push(HashMap::new());
        if let Some(init) = init {
            match init {
                ForInit::Declaration(d) => self.declaration(d)?,
                ForInit::Expression(Some(e)) => {
                    self.expr(e)?;
                }
                _ => {}
            }
        }
        let header = self.builder.create_block();
        let body_block = self.builder.create_block();
        let increment = self.builder.create_block();
        let exit = self.builder.create_block();
        self.jump(if do_first { body_block } else { header });
        self.builder.switch_to_block(header);
        if let Some(cond) = cond {
            self.branch(cond, body_block, exit)?;
        } else {
            self.jump(body_block);
        }
        self.breaks.push(exit);
        self.continues.push(increment);
        self.builder.switch_to_block(body_block);
        self.statement(body)?;
        self.jump(increment);
        self.builder.switch_to_block(increment);
        if let Some(step) = step {
            self.expr(step)?;
        }
        self.jump(header);
        self.breaks.pop();
        self.continues.pop();
        self.builder.switch_to_block(exit);
        self.scopes.pop();
        Ok(())
    }
    fn switch(&mut self, expr: &Expression, body: &Statement) -> Result<()> {
        fn collect<'a>(s: &'a Statement, cases: &mut Vec<Option<&'a Expression>>) {
            match s {
                Statement::Labeled(LabeledStatement::Case(e, s)) => {
                    cases.push(Some(e));
                    collect(s, cases);
                }
                Statement::Labeled(LabeledStatement::Default(s)) => {
                    cases.push(None);
                    collect(s, cases);
                }
                Statement::Compound(b) => {
                    for item in &b.items {
                        if let BlockItem::Statement(s) = item {
                            collect(s, cases);
                        }
                    }
                }
                Statement::Selection(SelectionStatement::If(_, a, b)) => {
                    collect(a, cases);
                    if let Some(b) = b {
                        collect(b, cases);
                    }
                }
                _ => {}
            }
        }
        let mut labels = Vec::new();
        collect(body, &mut labels);
        let value = self.expr(expr)?;
        let ty = value.ty.promoted();
        let value = self.cast(value, &ty)?;
        let exit = self.builder.create_block();
        let blocks: Vec<_> = labels.iter().map(|_| self.builder.create_block()).collect();
        let mut default = exit;
        let mut seen = HashSet::new();
        let mut has_default = false;
        let mut cases = Vec::new();
        let bits = match ty.plain() {
            CType::Int(bits, _) => *bits,
            _ => return fail("switch requires integer"),
        };
        let mask = u64::MAX >> (64 - bits);
        for (label, &block) in labels.iter().zip(&blocks) {
            if let Some(expr) = label {
                let value = self.types.constant(expr)? as u64 & mask;
                if !seen.insert(value) {
                    return fail("duplicate case after conversion");
                }
                cases.push((value, block));
            } else {
                if has_default {
                    return fail("duplicate default");
                }
                has_default = true;
                default = block;
            }
        }
        let range = cases
            .iter()
            .map(|&(v, _)| v)
            .min()
            .zip(cases.iter().map(|&(v, _)| v).max());
        if let Some((min, max)) = range
            && bits == 32
            && max - min < 128
            && (max - min + 1) as usize <= cases.len() * 3
        {
            let first = self.integer(min, ty.clone())?;
            let index = self.builder.ins().isub(value.value, first.value);
            let mut targets =
                vec![veloc_mir::SuccessorData::new(default, &[]); (max - min + 1) as usize];
            for (value, block) in cases {
                targets[(value - min) as usize] = veloc_mir::SuccessorData::new(block, &[]);
            }
            self.builder.ins().br_table(
                index,
                veloc_mir::SuccessorData::new(default, &[]),
                &targets,
            );
        } else {
            for (bits, block) in cases {
                let case = self.integer(bits, ty.clone())?;
                let cond = self.builder.ins().icmp(IntCC::Eq, value.value, case.value);
                let next = self.builder.create_block();
                self.builder.ins().br(cond, block, &[], next, &[]);
                self.builder.switch_to_block(next);
            }
            self.jump(default);
        }
        self.breaks.push(exit);
        self.cases.push(blocks.into());
        self.statement(body)?;
        self.jump(exit);
        self.cases.pop();
        self.breaks.pop();
        self.builder.switch_to_block(exit);
        Ok(())
    }
    pub(super) fn logical(
        &mut self,
        a: &Expression,
        b: &Expression,
        or: bool,
    ) -> Result<TypedValue> {
        let value = self.expr(a)?;
        let cond = self.truth(value)?;
        let rhs = self.builder.create_block();
        let short = self.builder.create_block();
        let merge = self.builder.create_block();
        let result = self.variable(&CType::INT)?;
        let (yes, no) = if or { (short, rhs) } else { (rhs, short) };
        self.builder.ins().br(cond, yes, &[], no, &[]);
        self.builder.switch_to_block(short);
        let value = self.integer(or as u64, CType::INT)?;
        self.builder.def_var(result, value.value);
        self.jump(merge);
        self.builder.switch_to_block(rhs);
        let value = self.expr(b)?;
        let cond = self.truth(value)?;
        let value = self.boolean_int(cond);
        self.builder.def_var(result, value.value);
        self.jump(merge);
        self.builder.switch_to_block(merge);
        Ok(TypedValue {
            value: self.builder.use_var(result),
            ty: CType::INT,
        })
    }
    pub(super) fn conditional(
        &mut self,
        c: &Expression,
        a: &Expression,
        b: &Expression,
    ) -> Result<TypedValue> {
        let ty = CType::common(&self.expr_type(a)?, &self.expr_type(b)?);
        let yes = self.builder.create_block();
        let no = self.builder.create_block();
        let merge = self.builder.create_block();
        let result = if ty == CType::Void {
            None
        } else {
            Some(self.variable(&ty)?)
        };
        self.branch(c, yes, no)?;
        for (block, e) in [(yes, a), (no, b)] {
            self.builder.switch_to_block(block);
            let value = self.expr(e)?;
            let value = self.cast(value, &ty)?;
            if let Some(result) = result {
                self.builder.def_var(result, value.value);
            }
            self.jump(merge);
        }
        self.builder.switch_to_block(merge);
        let value = if let Some(result) = result {
            self.builder.use_var(result)
        } else {
            self.integer(0, CType::INT)?.value
        };
        Ok(TypedValue { value, ty })
    }
}
