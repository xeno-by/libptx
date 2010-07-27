using System.Diagnostics;
using System.IO;
using Libptx.Common;
using Libptx.Common.Types.Pointers;
using Libptx.Expressions.Slots;
using Type=Libptx.Common.Types.Type;
using XenoGears.Assertions;

namespace Libptx.Expressions.Addresses
{
    // this table describes semantics of various Base/Offset combos:
    //
    //         |      null     |   null + off  |   reg + off   |   var + off   |
    // ========|===============|===============|===============|===============|
    // null    |    invalid    |     [off]     |  [reg + off]  |   var + off   |
    // var     |    invalid    |    arr[off]   | arr[reg + off]|    invalid    |
    // ========|===============|===============|===============|===============|

    [DebuggerNonUserCode]
    public partial class Address : Atom, Expression
    {
        public Expression Base { get; set; }
        public Offset Offset { get; set; }

        public Type Type
        {
            get { return typeof(Ptr); }
        }

        protected override void CustomValidate(Module ctx)
        {
            if (Base != null) Base.Validate(ctx);
            if (Offset != null) Offset.Validate(ctx);
            else throw AssertionHelper.Fail();

            if (Base != null)
            {
                (Base is Var && Base.is_arr()).AssertTrue();
                (Offset.Base == null || Offset.Base is Reg).AssertTrue();
            }
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            if (Base != null) writer.Write(Base);
            if (Offset != null) writer.Write(Offset);
        }
    }
}