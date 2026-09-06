// Shared by the runtime enum generator and build-time primitive binding generator.
define_generic_opcodes! {
    // ==================== 整数算术 ====================
    G_ADD => Add,  // 加法
    G_SUB => Sub,  // 减法
    G_MUL => Mul,  // 乘法
    G_SDIV, // 有符号除法
    G_UDIV, // 无符号除法
    G_SREM, // 有符号取余
    G_UREM, // 无符号取余
    G_NEG => Neg,  // 取负

    // ==================== 浮点算术 ====================
    G_FADD,  // 浮点加法
    G_FSUB,  // 浮点减法
    G_FMUL,  // 浮点乘法
    G_FDIV,  // 浮点除法
    G_FNEG,  // 浮点取负
    G_FABS,  // 浮点绝对值
    G_FSQRT, // 浮点开方

    // ==================== 位运算 ====================
    G_AND => And,   // 按位与
    G_OR => Or,    // 按位或
    G_XOR => Xor,   // 按位异或
    G_CTPOP, // 统计置位位数
    G_CTLZ,  // 统计前导零
    G_CTTZ,  // 统计尾随零
    G_SHL,   // 逻辑左移
    G_LSHR,  // 逻辑右移
    G_ASHR,  // 算术右移
    G_ROTL,  // 循环左移
    G_ROTR,  // 循环右移

    // ==================== 比较 ====================
    G_ICMP,            // 整数比较
    G_FCMP,            // 浮点比较
    G_IEQZ,            // 整数等于零比较 (Dst, Src)
    G_ANYEXT,          // 任意扩展
    G_ABS,             // 绝对值
    G_SMIN,            // 有符号最小值
    G_SMAX,            // 有符号最大值
    G_UMIN,            // 无符号最小值
    G_UMAX,            // 无符号最大值
    G_UADDO,           // 无符号加法溢出
    G_SADDO,           // 有符号加法溢出
    G_USUBO,           // 无符号减法溢出
    G_SSUBO,           // 有符号减法溢出
    G_UADDE,           // 带进位无符号加法
    G_SADDE,           // 带进位有符号加法
    G_USUBE,           // 带借位无符号减法
    G_SSUBE,           // 带借位有符号减法
    G_UMULO,           // 无符号乘法溢出
    G_SMULO,           // 有符号乘法溢出
    G_UMULH,           // 无符号乘法高位
    G_SMULH,           // 有符号乘法高位
    G_CTLZ_ZERO_UNDEF, // 计数前导零，零输入未定义
    G_CTTZ_ZERO_UNDEF, // 计数尾随零，零输入未定义
    G_SADDSAT,         // 有符号饱和加法
    G_UADDSAT,         // 无符号饱和加法
    G_SSUBSAT,         // 有符号饱和减法
    G_USUBSAT,         // 无符号饱和减法

    // ==================== 内存操作 ====================
    G_LOAD,        // 加载 (Dst, Ptr)
    G_STORE,       // 存储 (Src, Ptr)
    G_PTR_ADD,     // 指针加法 (Base, Offset)
    G_STACK_LOAD,  // 从栈加载
    G_STACK_STORE, // 存储到栈
    G_STACK_ADDR,  // 获取栈地址

    G_OFFSET_LOAD,   // [Dst] = Load (Base + Offset)
    G_OFFSET_STORE,  // [Base + Offset] = Store (Src)
    G_INDEXED_LOAD,  // 带写回的加载 (Dst, Base_out, Base_in, Offset)
    G_INDEXED_STORE, // 带写回的存储 (Base_out, Src, Base_in, Offset)

    // ==================== 常量 ====================
    G_CONSTANT,  // 整数常量
    G_FCONSTANT, // 浮点常量

    // ==================== 类型转换 ====================
    G_TRUNC,   // 截断
    G_ZEXT,    // 零扩展
    G_SEXT,    // 符号扩展
    G_FPTOSI,  // 浮点到有符号整数
    G_FPTOUI,  // 浮点到无符号整数
    G_SITOFP,  // 有符号整数到浮点
    G_UITOFP,  // 无符号整数到浮点
    G_FPTRUNC, // 浮点截断
    G_FPEXT,   // 浮点扩展
    G_BITCAST, // 位转换

    G_INTTOPTR, // 整数转指针
    G_PTRTOINT, // 指针转整数

    // ==================== 控制流 ====================
    G_BR,      // 无条件跳转
    G_BRCOND,  // 条件跳转
    G_BRIND,   // 间接跳转
    G_BRJT,    // 跳转表
    G_RET,     // 返回
    G_CALL,    // 直接调用
    G_CALLIND, // 间接调用
    G_ARG,     // 获取函数参数 (Index)

    // ==================== 其他 ====================
    G_SELECT,  // 选择
    G_COPY,    // 寄存器拷贝
    G_PHI,     // PHI 节点
    G_EXTRACT, // 提取向量元素
    G_INSERT,  // 插入向量元素
    G_UNMERGE, // 拆分数值
    G_MERGE,   // 合并数值

    // ==================== 平台相关（需 Lower）====================
    G_READCYCLECOUNTER, // 读取周期计数器
    G_UNREACHABLE,      // 不可达
}
