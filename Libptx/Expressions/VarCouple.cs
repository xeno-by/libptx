using System;
using System.IO;

namespace Libptx.Expressions
{
    public class VarCouple : Var
    {
        public Var Fst { get; set; }
        public Var Snd { get; set; }

        protected override void CustomValidate(Module ctx)
        {
            throw new NotImplementedException();
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            throw new NotImplementedException();
        }
    }
}