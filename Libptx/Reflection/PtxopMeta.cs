using System;
using System.Collections.ObjectModel;
using System.Diagnostics;
using Libptx.Expressions;
using Libptx.Instructions;

namespace Libptx.Reflection
{
    [DebuggerNonUserCode]
    public class PtxopMeta
    {
        public ptxop Ptxop { get; private set; }
        public PtxopSig Sig { get { throw new NotImplementedException(); } }

        public String Opcode { get; private set; }
        public ReadOnlyCollection<Object> Mods { get; private set; }
        public ReadOnlyCollection<Object> Affixes { get; private set; }
        public ReadOnlyCollection<Expression> Operands { get; private set; }

        public PtxopMeta(ptxop ptxop)
        {
            Ptxop = ptxop;

            throw new NotImplementedException();
        }
    }
}