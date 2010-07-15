using System;
using System.Collections.Generic;
using Libcuda.Versions;
using Libptx.Expressions;
using Libptx.Statements;
using XenoGears.Assertions;

namespace Libptx
{
    public class Func : Block, Callable
    {
        public virtual String Name { get; set; }
        public virtual IList<Var> Params { get; private set; }
        public virtual IList<Var> Rets { get; private set; }

        public override void Validate()
        {
            var abi2 = Ctx.Target >= HardwareIsa.SM_20 && Ctx.Version >= SoftwareIsa.PTX_20;
            Params.AssertEach(p => p.Space == reg || (abi2 && p.Space == param));
        }
    }
}