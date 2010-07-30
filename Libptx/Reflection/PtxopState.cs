using System;
using System.Collections.ObjectModel;
using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Expressions;
using Libptx.Instructions;
using XenoGears.Assertions;
using XenoGears.Reflection.Shortcuts;
using System.Linq;
using XenoGears.Reflection.Attributes;
using XenoGears.Functional;

namespace Libptx.Reflection
{
    [DebuggerNonUserCode]
    public class PtxopState
    {
        public ptxop Ptxop { get; private set; }
        public PtxopSig Sig { get { throw new NotImplementedException(); } }

        public String Opcode { get; private set; }
        public ReadOnlyCollection<Object> Mods { get; private set; }
        public ReadOnlyCollection<Object> Affixes { get; private set; }
        public ReadOnlyCollection<Expression> Operands { get; private set; }

        internal PtxopState(ptxop ptxop)
        {
            Ptxop = ptxop;

            // todo. implement Sig (i.e. find out the exact Sig that corresponds to current state of ptxop)
            Opcode = ptxop.PtxopSigs().AssertFirst().Opcode;

            // todo. when Sig is implemented, use only such properties and in such order that are mentioned in Sig
            var props = ptxop.GetType().GetProperties(BF.PublicInstance);
            Mods = props.Where(p => p.HasAttr<ModAttribute>()).Select(p => p.GetValue(ptxop, null)).ToReadOnly();
            Affixes = props.Where(p => p.HasAttr<AffixAttribute>()).Select(p => p.GetValue(ptxop, null)).ToReadOnly();
            Operands = props.Where(p => typeof(Expression).IsAssignableFrom(p.PropertyType)).Select(p => p.GetValue(ptxop, null).AssertCast<Expression>()).ToReadOnly();
        }
    }
}