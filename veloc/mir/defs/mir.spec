import "formats.spec";

// Logical signatures name SSA operands and properties; result names are optional.
// Storage mappings preserve the current compact MIR layout and SSA traversal order.
// Structural constraints compile to direct checks, not a runtime rule list.

// First-class callable values. Captures bind a prefix of the function parameters;
// the resulting callable signature describes the remaining inputs and answer.
// Allocation/ownership are effects: these operations are not arithmetic CSE/DCE.
op ClosureNew(func_id: FuncId, move captures: sequence(Value), cleanup: FuncId) -> (result: Value<Callable>) {
    memory = all;
    mnemonic = "closure-new";
    storage = ClosureNew { func_id, captures, cleanup };
    text = "{func_id}({captures}) : {function(func_id)}, {cleanup}";

    verify(ctx: VerifyContext) {
        let func = ctx.function_signature(func_id)?;
        let sig = ctx.signature(result.signature()?)?;
        let cleanup_sig = ctx.function_signature(cleanup)?;
        require(func.is_variadic() == sig.is_variadic(), "callable variadic signature mismatch");
        require(!cleanup_sig.is_variadic(), "cleanup must have a fixed signature");
        require(result.is_owned(), "callable ownership kind mismatch");
        require(func.same_call_conv(sig), "callable calling convention mismatch");
        require(matches_types(captures, prefix(func.params(), len(captures))), "capture parameter type mismatch");
        require(suffix(func.params(), len(captures)) == sig.params() && func.returns() == sig.returns(), "callable inputs or answer do not match the unbound function signature");
        require(all(captures, |v| !v.ty().is_local() && !v.ty().is_ptr()), "capture would escape a borrow or duplicate an owned environment");
        require(cleanup_sig.params() == prefix(func.params(), len(captures)) && len(cleanup_sig.returns()) == 0, "cleanup must consume the captured parameters and return void");
    }
}

op ClosureLocal(func_id: FuncId, captures: sequence(Value)) -> (result: Value<Callable>) {
    memory = all;
    mnemonic = "closure-local";
    storage = Closure { func_id, captures };
    text = "{func_id}({captures}) : {function(func_id)}";

    verify(ctx: VerifyContext) {
        let func = ctx.function_signature(func_id)?;
        let sig = ctx.signature(result.signature()?)?;
        require(result.is_local(), "callable ownership kind mismatch");
        require(func.same_call_conv(sig), "callable calling convention mismatch");
        require(func.is_variadic() == sig.is_variadic(), "callable variadic signature mismatch");
        require(matches_types(captures, prefix(func.params(), len(captures))), "capture parameter type mismatch");
        require(suffix(func.params(), len(captures)) == sig.params() && func.returns() == sig.returns(), "callable inputs or answer do not match the unbound function signature");
        require(all(captures, |v| !v.ty().is_owned()), "capture would escape a borrow or duplicate an owned environment");
    }
}

op ClosureShared(func_id: FuncId, captures: sequence(Value)) -> (result: Value<Callable>) {
    memory = all;
    mnemonic = "closure-shared";
    storage = Closure { func_id, captures };
    text = "{func_id}({captures}) : {function(func_id)}";

    verify(ctx: VerifyContext) {
        let func = ctx.function_signature(func_id)?;
        let sig = ctx.signature(result.signature()?)?;
        require(result.is_shared(), "callable ownership kind mismatch");
        require(func.same_call_conv(sig), "callable calling convention mismatch");
        require(func.is_variadic() == sig.is_variadic(), "callable variadic signature mismatch");
        require(matches_types(captures, prefix(func.params(), len(captures))), "capture parameter type mismatch");
        require(suffix(func.params(), len(captures)) == sig.params() && func.returns() == sig.returns(), "callable inputs or answer do not match the unbound function signature");
        require(all(captures, |v| (v.ty().is_compact() && !v.ty().is_ptr()) || v.ty().is_shared()), "capture would escape a borrow or duplicate an owned environment");
    }
}

op TailCall(func_id: FuncId, move args: sequence(Value)) -> () {
    traps = callee(func_id);
    memory = callee(func_id);
    control = terminator;
    mnemonic = "tail-call";
    storage = TailCall { func_id, args };
    text = "{func_id}({args}) : {function(func_id)}";

    verify(ctx: VerifyContext) {
        let func = ctx.function_signature(func_id)?;
        require(len(args) >= len(func.params()) && (func.is_variadic() || len(args) == len(func.params())), "tail-call argument count mismatch");
        require(matches_types(prefix(args, len(func.params())), func.params()), "tail-call parameter type mismatch");
        require(func.returns() == ctx.current_signature()?.returns(), "tail-call answer type mismatch");
    }
}

op TailCallValue(move callee: Value<Callable>, move args: sequence(Value)) -> () {
    traps = possible;
    memory = all;
    control = terminator;
    mnemonic = "tail-call-value";
    storage = CallValue { callee, args };
    text = "{callee}({args})";

    verify(ctx: VerifyContext) {
        let callee_sig = ctx.signature(callee.ty().signature()?)?;
        require(len(args) >= len(callee_sig.params()) && (callee_sig.is_variadic() || len(args) == len(callee_sig.params())), "tail-call-value argument count mismatch");
        require(matches_types(prefix(args, len(callee_sig.params())), callee_sig.params()), "tail-call-value parameter type mismatch");
        require(callee_sig.returns() == ctx.current_signature()?.returns(), "tail-call-value answer type mismatch");
    }
}

op CallValue(move callee: Value<Callable>, move args: sequence(Value)) -> signature {
    traps = possible;
    memory = all;
    mnemonic = "call-value";
    storage = CallValue { callee, args };
    signature = callable(callee);
    text = "{callee}({args})";
}

op ClosureDrop(move callee: Value<Callable>) -> () {
    traps = possible;
    memory = all;
    mnemonic = "closure-drop";
    storage = Unary { arg: callee };

    verify {
        require(callee.ty().is_owned(), "only an owned one-shot callable can be dropped");
    }
}

op IAdd<T: Integer>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    mnemonic = "iadd";
    storage = Binary { args: [lhs, rhs] };
    semantics = bv.add(lhs, rhs)
; }

op ISub<T: Integer>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    mnemonic = "isub";
    storage = Binary { args: [lhs, rhs] };
    semantics = bv.sub(lhs, rhs)
; }

op IMul<T: Integer>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    mnemonic = "imul";
    storage = Binary { args: [lhs, rhs] };
    semantics = bv.mul(lhs, rhs)
; }

op INeg<T: Integer>(arg: Value<T>) -> Value<T> {
    mnemonic = "ineg";
    storage = Unary { arg };
    semantics = bv.sub(bv.zero(), arg)
; }

// Signed saturation: clamp to the signed range of each lane.
op IAddSatS<T: Integer>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    mnemonic = "iadd-sat-s";
    storage = Binary { args: [lhs, rhs] };
    semantics = select(
        bv.slt(bv.and(bv.xor(lhs, bv.add(lhs, rhs)), bv.xor(rhs, bv.add(lhs, rhs))), bv.zero()),
        select(bv.slt(lhs, bv.zero()), bv.smin(), bv.xor(bv.smin(), bv.ones())),
        bv.add(lhs, rhs)
    )
; }

op ISubSatS<T: Integer>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    mnemonic = "isub-sat-s";
    storage = Binary { args: [lhs, rhs] };
    semantics = select(
        bv.slt(bv.and(bv.xor(lhs, rhs), bv.xor(lhs, bv.sub(lhs, rhs))), bv.zero()),
        select(bv.slt(lhs, bv.zero()), bv.smin(), bv.xor(bv.smin(), bv.ones())),
        bv.sub(lhs, rhs)
    )
; }

op IAddWithOverflow<T: ScalarInteger>(lhs: Value<T>, rhs: Value<T>) -> (Value<T>, Value<Type::BOOL>) {
    mnemonic = "iadd-with-overflow";
    storage = Binary { args: [lhs, rhs] };
    // The result wraps; the flag reports SIGNED overflow, not unsigned carry.
    semantics = [
        bv.add(lhs, rhs),
        bv.slt(bv.and(bv.xor(lhs, bv.add(lhs, rhs)), bv.xor(rhs, bv.add(lhs, rhs))), bv.zero())
    ]
; }

op ISubWithOverflow<T: ScalarInteger>(lhs: Value<T>, rhs: Value<T>) -> (Value<T>, Value<Type::BOOL>) {
    mnemonic = "isub-with-overflow";
    storage = Binary { args: [lhs, rhs] };
    semantics = [
        bv.sub(lhs, rhs),
        bv.slt(bv.and(bv.xor(lhs, rhs), bv.xor(lhs, bv.sub(lhs, rhs))), bv.zero())
    ]
; }

op IMulWithOverflow<T: ScalarInteger>(lhs: Value<T>, rhs: Value<T>) -> (Value<T>, Value<Type::BOOL>) {
    mnemonic = "imul-with-overflow";
    storage = Binary { args: [lhs, rhs] };
    // bv.sdiv is total; this does not introduce a division trap.
    semantics = [
        bv.mul(lhs, rhs),
        bv.and(bv.xor(bv.eq(rhs, bv.zero()), bv.ones(result(1))),
            bv.or(bv.xor(bv.eq(bv.sdiv(bv.mul(lhs, rhs), rhs), lhs), bv.ones(result(1))),
                bv.and(bv.eq(lhs, bv.smin()), bv.eq(rhs, bv.ones()))))
    ]
; }

// Division and remainder require a nonzero divisor. The signed forms also
// exclude signed MIN and -1 as an operand pair. Invalid inputs are UB.
op IDivS<T: Integer>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    mnemonic = "idiv-s";
    storage = Binary { args: [lhs, rhs] };
    semantics = bv.sdiv(lhs, rhs);
}

op IDivU<T: Integer>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    mnemonic = "idiv-u";
    storage = Binary { args: [lhs, rhs] };
    semantics = bv.udiv(lhs, rhs);
}

op IRemS<T: Integer>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    mnemonic = "irem-s";
    storage = Binary { args: [lhs, rhs] };
    semantics = bv.srem(lhs, rhs);
}

op IRemU<T: Integer>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    mnemonic = "irem-u";
    storage = Binary { args: [lhs, rhs] };
    semantics = bv.urem(lhs, rhs);
}

op FAdd<T: Float>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    mnemonic = "fadd";
    storage = Binary { args: [lhs, rhs] };
}

op FSub<T: Float>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    mnemonic = "fsub";
    storage = Binary { args: [lhs, rhs] };
}

op FMul<T: Float>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    mnemonic = "fmul";
    storage = Binary { args: [lhs, rhs] };
}

op FNeg<T: Float>(arg: Value<T>) -> Value<T> {
    mnemonic = "fneg";
    storage = Unary { arg };
}

op FDiv<T: Float>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    mnemonic = "fdiv";
    storage = Binary { args: [lhs, rhs] };
}

op FMin<T: Float>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    mnemonic = "fmin";
    storage = Binary { args: [lhs, rhs] };
}

op FMax<T: Float>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    mnemonic = "fmax";
    storage = Binary { args: [lhs, rhs] };
}

op FCopysign<T: Float>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    mnemonic = "fcopysign";
    storage = Binary { args: [lhs, rhs] };
}

op FAbs<T: Float>(arg: Value<T>) -> Value<T> {
    mnemonic = "fabs";
    storage = Unary { arg };
}

op FSqrt<T: Float>(arg: Value<T>) -> Value<T> {
    mnemonic = "fsqrt";
    storage = Unary { arg };
}

op FCeil<T: Float>(arg: Value<T>) -> Value<T> {
    mnemonic = "fceil";
    storage = Unary { arg };
}

op FFloor<T: Float>(arg: Value<T>) -> Value<T> {
    mnemonic = "ffloor";
    storage = Unary { arg };
}

op FTrunc<T: Float>(arg: Value<T>) -> Value<T> {
    mnemonic = "ftrunc";
    storage = Unary { arg };
}

op FNearest<T: Float>(arg: Value<T>) -> Value<T> {
    mnemonic = "fnearest";
    storage = Unary { arg };
}

op IAnd<T: Integer | Type::BOOL | vectors(Type::BOOL)>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    mnemonic = "iand";
    storage = Binary { args: [lhs, rhs] };
    semantics = bv.and(lhs, rhs)
; }

op IOr<T: Integer | Type::BOOL | vectors(Type::BOOL)>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    mnemonic = "ior";
    storage = Binary { args: [lhs, rhs] };
    semantics = bv.or(lhs, rhs)
; }

op IXor<T: Integer | Type::BOOL | vectors(Type::BOOL)>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    mnemonic = "ixor";
    storage = Binary { args: [lhs, rhs] };
    semantics = bv.xor(lhs, rhs)
; }

op IShl<T: Integer>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    mnemonic = "ishl";
    storage = Binary { args: [lhs, rhs] };
    semantics = bv.shl(lhs, bv.urem(rhs, bv.width()))
; }

op IShrS<T: Integer>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    mnemonic = "ishr-s";
    storage = Binary { args: [lhs, rhs] };
    semantics = bv.ashr(lhs, bv.urem(rhs, bv.width()))
; }

op IShrU<T: Integer>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    mnemonic = "ishr-u";
    storage = Binary { args: [lhs, rhs] };
    semantics = bv.lshr(lhs, bv.urem(rhs, bv.width()))
; }

op IRotl<T: Integer>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    mnemonic = "irotl";
    storage = Binary { args: [lhs, rhs] };
    semantics = bv.or(
        bv.shl(lhs, bv.urem(rhs, bv.width())),
        bv.lshr(lhs, bv.sub(bv.width(), bv.urem(rhs, bv.width())))
    )
; }

op IRotr<T: Integer>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {
    mnemonic = "irotr";
    storage = Binary { args: [lhs, rhs] };
    semantics = bv.or(
        bv.lshr(lhs, bv.urem(rhs, bv.width())),
        bv.shl(lhs, bv.sub(bv.width(), bv.urem(rhs, bv.width())))
    )
; }

op IClz<T: Integer>(arg: Value<T>) -> Value<T> {
    mnemonic = "iclz";
    storage = Unary { arg };
    semantics = bv.clz(arg)
; }

op ICtz<T: Integer>(arg: Value<T>) -> Value<T> {
    mnemonic = "ictz";
    storage = Unary { arg };
    semantics = bv.ctz(arg)
; }

op IPopcnt<T: Integer>(arg: Value<T>) -> Value<T> {
    mnemonic = "ipopcnt";
    storage = Unary { arg };
    semantics = bv.popcnt(arg)
; }

op IEqz(arg: Value<ScalarInteger>) -> Value<Type::BOOL> {
    mnemonic = "ieqz";
    storage = Unary { arg };
    semantics = bv.eq(arg, bv.zero())
; }

op Icmp<T: ScalarInteger | Type::PTR>(kind: IntCC, lhs: Value<T>, rhs: Value<T>) -> Value<Type::BOOL> {
    mnemonic = "icmp";
    storage = IntCompare { kind, args: [lhs, rhs] };
    text = "{kind} {lhs}, {rhs}";
    semantics = bv.cmp(kind, lhs, rhs);
    verify {
        require(!lhs.ty().is_ptr() || kind == IntCC::Eq || kind == IntCC::Ne, "pointer comparison only supports eq and ne");
    }
}

op Fcmp<T: ScalarFloat>(kind: FloatCC, lhs: Value<T>, rhs: Value<T>) -> Value<Type::BOOL> {
    mnemonic = "fcmp";
    storage = FloatCompare { kind, args: [lhs, rhs] };
    text = "{kind} {lhs}, {rhs}";
}

op ExtendS<T: Integer, U: Integer>(arg: Value<T>) -> Value<U> {
    mnemonic = "extends";
    storage = Unary { arg };
    verify {
        require(U.same_shape(T), "input and result must have the same shape");
        require(U.wider_than(T), "result must have more bits per lane than arg");
    }
    semantics = bv.sext(arg, result(0))
; }

op ExtendU<T: Integer | Type::BOOL | vectors(Type::BOOL), U: Integer>(arg: Value<T>) -> Value<U> {
    mnemonic = "extendu";
    storage = Unary { arg };
    verify {
        require(U.same_shape(T), "input and result must have the same shape");
        require(U.wider_than(T), "result must have more bits per lane than arg");
    }
    semantics = bv.zext(arg, result(0))
; }

op Wrap<T: Integer, U: Integer>(arg: Value<T>) -> Value<U> {
    mnemonic = "wrap";
    storage = Unary { arg };
    verify {
        require(U.same_shape(T), "input and result must have the same shape");
        require(T.wider_than(U), "result must have fewer bits per lane than arg");
    }
    semantics = bv.trunc(arg, result(0))
; }

op FloatToIntSatS<T: Float, U: Integer>(arg: Value<T>) -> Value<U> {
    mnemonic = "float-to-int-sat-s";
    storage = Unary { arg };
    verify {
        require(U.same_shape(T), "input and result must have the same shape");
    }
}

op FloatToIntSatU<T: Float, U: Integer>(arg: Value<T>) -> Value<U> {
    mnemonic = "float-to-int-sat-u";
    storage = Unary { arg };
    verify {
        require(U.same_shape(T), "input and result must have the same shape");
    }
}

op FloatToIntS<T: Float, U: Integer>(arg: Value<T>) -> Value<U> {
    mnemonic = "float-to-int-s";
    storage = Unary { arg };
    verify {
        require(U.same_shape(T), "input and result must have the same shape");
    }
}

op FloatToIntU<T: Float, U: Integer>(arg: Value<T>) -> Value<U> {
    mnemonic = "float-to-int-u";
    storage = Unary { arg };
    verify {
        require(U.same_shape(T), "input and result must have the same shape");
    }
}

op IntToFloatS<T: Integer, U: Float>(arg: Value<T>) -> Value<U> {
    mnemonic = "int-to-float-s";
    storage = Unary { arg };
    verify {
        require(U.same_shape(T), "input and result must have the same shape");
    }
}

op IntToFloatU<T: Integer, U: Float>(arg: Value<T>) -> Value<U> {
    mnemonic = "int-to-float-u";
    storage = Unary { arg };
    verify {
        require(U.same_shape(T), "input and result must have the same shape");
    }
}

op FloatPromote(arg: Value<Type::F32>) -> Value<Type::F64> {
    mnemonic = "float-promote";
    storage = Unary { arg };
}

op FloatDemote(arg: Value<Type::F64>) -> Value<Type::F32> {
    mnemonic = "float-demote";
    storage = Unary { arg };
}

op Reinterpret(arg: Value<Number>) -> (result: Value<Number>) {
    mnemonic = "reinterpret";
    storage = Unary { arg };
    verify {
        require(arg.ty() != result && arg.ty().bit_size()? == result.bit_size()?, "reinterpret requires distinct types with equal bit sizes");
    }
}

op IntToPtr(arg: Value<ScalarInteger>) -> Value<Type::PTR> {
    mnemonic = "inttoptr";
    storage = IntToPtr { arg };
}

op PtrToInt(arg: Value<Type::PTR>) -> Value<ScalarInteger> {
    mnemonic = "ptrtoint";
    storage = PtrToInt { arg };
}

// Ordinary memory accesses require a valid address for each accessed byte.
op Load(ptr: Value<Type::PTR>, offset: u32, flags: MemFlags) -> (result: Value<Any>) {
    memory = read;
    mnemonic = "load";
    storage = Load { ptr, offset, flags };
    text = "{.flags} {ptr}, offset={offset}";
}

op Store(ptr: Value<Type::PTR>, value: Value<Any>, offset: u32, flags: MemFlags) -> () {
    memory = write;
    mnemonic = "store";
    storage = Store { ptr, value, offset, flags };
    text = "{.flags} {value}, {ptr}, offset={offset}";
}

op GlobalAddr(global: GlobalId) -> Value<Type::PTR> {
    mnemonic = "global-addr";
    storage = GlobalAddr { global };
    text = "{global}";
    verify(ctx: VerifyContext) {
        require(ctx.has_global(global), "unknown global");
    }
}

op FuncAddr(func_id: FuncId) -> Value<Type::PTR> {
    mnemonic = "func-addr";
    storage = FuncAddr { func_id };
    text = "{func_id}";
    verify(ctx: VerifyContext) {
        let signature = ctx.function_signature(func_id)?;
    }
}

op PtrOffset(ptr: Value<Type::PTR>, offset: i32) -> Value<Type::PTR> {
    mnemonic = "ptr-offset";
    storage = PtrOffset { ptr, offset };
}

op PtrIndex(ptr: Value<Type::PTR>, index: Value<ScalarInteger>, imm: PtrIndexImm) -> Value<Type::PTR> {
    mnemonic = "ptr-index";
    storage = PtrIndex { ptr, index, imm_id: imm };
    text = "{ptr}, {index}, scale={imm.scale}, offset={imm.offset}";

    verify {
        require(imm.scale != 0, "ptr-index scale must be non-zero");
    }
}

op Call(func_id: FuncId, move args: sequence(Value)) -> signature {
    traps = callee(func_id);
    memory = callee(func_id);
    mnemonic = "call";
    storage = Call { func_id, args };
    signature = function(func_id);
    text = "{func_id}({args}) : {function(func_id)}";
}


op CallIndirect(sig_id: SigId, ptr: Value<Type::PTR>, move args: sequence(Value)) -> signature {
    traps = possible;
    memory = all;
    mnemonic = "call-indirect";
    storage = CallIndirect { ptr, args, sig_id };
    signature = sig_id;
    text = "{ptr}({args}) : {sig_id}";

    verify(ctx: VerifyContext) {
        let sig = ctx.signature(sig_id)?;
        require(all(sig.types(), |t| !t.is_callable()), "indirect and intrinsic callable calls require an ownership-aware callable type");
    }
}

op CallIntrinsic(intrinsic: Intrinsic, sig_id: SigId, move args: sequence(Value)) -> signature {
    traps = possible;
    memory = all;
    mnemonic = "call-intrinsic";
    storage = CallIntrinsic { intrinsic, args, sig_id };
    signature = sig_id;
    text = "{intrinsic}({args}) : {sig_id}";

    verify(ctx: VerifyContext) {
        let sig = ctx.signature(sig_id)?;
        require(!sig.is_variadic(), "intrinsic calls require a fixed signature");
        require(all(sig.types(), |t| !t.is_callable()), "indirect and intrinsic callable calls require an ownership-aware callable type");
    }
}

op Jump(dest: successor) -> () {
    control = terminator;
    mnemonic = "jump";
    storage = Jump { dest };
}

op Br(condition: Value<Type::BOOL>, then_dest: successor, else_dest: successor) -> () {
    control = terminator;
    mnemonic = "br";
    storage = Br { condition, then_dest, else_dest };
}

op BrTable(index: Value<Type::I32>, cases: successors, default: successor) -> () {
    control = terminator;
    mnemonic = "br-table";
    storage = BrTable { index, table: table(cases, default) };
}

op Return(move values: sequence(Value)) -> () {
    control = terminator;
    mnemonic = "return";
    storage = Return { values };

    verify(ctx: VerifyContext) {
        require(matches_types(values, ctx.current_signature()?.returns()), "return value type mismatch");
    }
}

op Select<T: Any>(condition: Value<Type::BOOL>, if_true: Value<T>, if_false: Value<T>) -> Value<T> {
    mnemonic = "select";
    storage = Ternary { args: [condition, if_true, if_false] };
}

op Unreachable() -> () {
    control = abort;
    mnemonic = "unreachable";
    storage = Unreachable {};
}

op Nop() -> () {
    mnemonic = "nop";
    storage = Nop {};
}

op Splat<T: Scalar>(arg: Value<T>) -> Value<vector(T)> {
    mnemonic = "splat";
    storage = Unary { arg };
}

op Shuffle<T: Vector, M: Integer & Vector, R: Vector>(lhs: Value<T>, rhs: Value<T>, mask: Value<M>) -> Value<R> {
    mnemonic = "shuffle";
    storage = Ternary { args: [lhs, rhs, mask] };

    verify(ctx: VerifyContext) {
        require(T.is_fixed() && M.is_fixed() && R.is_fixed(), "shuffle requires fixed-width vectors");
        require(T.element_type()? == R.element_type()?, "shuffle input and result element types must match");
        require(M.lanes()? == R.lanes()?, "shuffle mask length must match its result lane count");
        let indices = ctx.vector_constant(mask)?;
        require(indices.unsigned_max()? < 2 * u64(lhs.ty().lanes()?), "shuffle selector is out of range");
    }
}

op InsertElement<T: Vector>(vector: Value<T>, scalar: Value<element(T)>, index: Value<ScalarInteger>) -> Value<T> {
    mnemonic = "insertelement";
    storage = Ternary { args: [vector, scalar, index] };
    // The unsigned index must be less than the runtime lane count.
}

op ExtractElement<T: Vector>(vector: Value<T>, index: Value<ScalarInteger>) -> Value<element(T)> {
    mnemonic = "extractelement";
    storage = Binary { args: [vector, index] };
    // The unsigned index must be less than the runtime lane count.
}

// Add in increasing lane order, starting at lane 0 (no extra zero).
// Integer addition wraps. Floating addition uses the scalar FAdd contract.
op ReduceAdd<T: Number & Vector>(arg: Value<T>) -> Value<element(T)> {
    mnemonic = "reduce-add";
    storage = Unary { arg };
}

// Integer extrema use signed comparison. Float extrema propagate NaN;
// min(-0,+0) is -0, max(-0,+0) is +0. Fold in increasing lane order.
op ReduceMin<T: Number & Vector>(arg: Value<T>) -> Value<element(T)> {
    mnemonic = "reduce-min";
    storage = Unary { arg };
}

op ReduceMax<T: Number & Vector>(arg: Value<T>) -> Value<element(T)> {
    mnemonic = "reduce-max";
    storage = Unary { arg };
}

op ReduceAnd<T: Integer & Vector | vectors(Type::BOOL)>(arg: Value<T>) -> Value<element(T)> {
    mnemonic = "reduce-and";
    storage = Unary { arg };
}

op ReduceOr<T: Integer & Vector | vectors(Type::BOOL)>(arg: Value<T>) -> Value<element(T)> {
    mnemonic = "reduce-or";
    storage = Unary { arg };
}

op ReduceXor<T: Integer & Vector | vectors(Type::BOOL)>(arg: Value<T>) -> Value<element(T)> {
    mnemonic = "reduce-xor";
    storage = Unary { arg };
}

op LoadStride(ptr: Value<Type::PTR>, stride: Value<ScalarInteger>, mem: VectorMemOptions) -> (result: Value<Vector>) {
    memory = read;
    mnemonic = "load-stride";
    storage = VectorLoadStrided { ptr, stride, ext: mem };
    text = "{mem.scale=1} {.mem.flags} {ptr}, stride={stride}, offset={mem.offset}[, mask={mem.mask}][, evl={mem.evl}]";

    verify {
        require(all(mem.mask, |v| v.ty().is_predicate() && v.ty().shape()? == result.shape()?), "mask must match vector shape");
        require(all(mem.evl, |v| v.ty() == Type::I32), "EVL must be i32");
    }
}

op StoreStride(ptr: Value<Type::PTR>, stride: Value<ScalarInteger>, value: Value<Vector>, mem: VectorMemOptions) -> () {
    memory = write;
    mnemonic = "store-stride";
    storage = VectorStoreStrided { args: [ptr, stride, value], ext: mem };
    text = "{mem.scale=1} {.mem.flags} {value}, {ptr}, stride={stride}, offset={mem.offset}[, mask={mem.mask}][, evl={mem.evl}]";

    verify {
        require(all(mem.mask, |v| v.ty().is_predicate() && v.ty().shape()? == value.ty().shape()?), "mask must match vector shape");
        require(all(mem.evl, |v| v.ty() == Type::I32), "EVL must be i32");
    }
}

op Gather<T: Integer & Vector, U: Vector>(ptr: Value<Type::PTR>, index: Value<T>, mem: VectorMemOptions) -> Value<U> {
    memory = read;
    mnemonic = "gather";
    storage = VectorGather { ptr, index, ext: mem };
    text = "{.mem.flags} {ptr}, index={index}, scale={mem.scale}, offset={mem.offset}[, mask={mem.mask}][, evl={mem.evl}]";

    verify {
        require(U.same_shape(T), "index and result must have the same shape");
        require(all(mem.mask, |v| v.ty().is_predicate() && v.ty().shape()? == U.shape()?), "mask must match vector shape");
        require(all(mem.evl, |v| v.ty() == Type::I32), "EVL must be i32");
    }
}

op Scatter<T: Integer & Vector, U: Vector>(ptr: Value<Type::PTR>, index: Value<T>, value: Value<U>, mem: VectorMemOptions) -> () {
    memory = write;
    mnemonic = "scatter";
    storage = VectorScatter { args: [ptr, index, value], ext: mem };
    text = "{.mem.flags} {value}, {ptr}, index={index}, scale={mem.scale}, offset={mem.offset}[, mask={mem.mask}][, evl={mem.evl}]";

    verify {
        require(U.same_shape(T), "index and value must have the same shape");
        require(all(mem.mask, |v| v.ty().is_predicate() && v.ty().shape()? == value.ty().shape()?), "mask must match vector shape");
        require(all(mem.evl, |v| v.ty() == Type::I32), "EVL must be i32");
    }
}
