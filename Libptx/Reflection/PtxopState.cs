using System;
using System.Diagnostics;
using System.Reflection;
using Libptx.Common.Annotations.Quanta;
using Libptx.Expressions;
using Libptx.Instructions;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;
using XenoGears.Collections.Dictionaries;
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

        public Expression Guard { get; private set; }
        public String Opcode { get; private set; }
        public OrderedDictionary<PropertyInfo, Object> Mods { get; private set; }
        public OrderedDictionary<PropertyInfo, Object> Affixes { get; private set; }
        public OrderedDictionary<PropertyInfo, Expression> Operands { get; private set; }
        public Tuple<PropertyInfo, Expression> Destination { get; private set; }

        internal PtxopState(ptxop ptxop)
        {
            Ptxop = ptxop;
            Guard = ptxop.Guard;

            // todo. implement Sig (i.e. find out the exact Sig that corresponds to current state of ptxop)
            Opcode = ptxop.Ptxopcode();

            Func<Object, PropertyInfo, Object> get_value = (o, p) =>
            {
                var v = p.GetValue(o, null);
                var is_default = Equals(v, p.PropertyType.IsValueType ? Activator.CreateInstance(p.PropertyType) : null);
                return is_default ? null : v;
            };

            // todo. when Sig is implemented, use only such properties and in such order that are mentioned in Sig
            var props = ptxop.GetType().GetProperties(BF.PublicInstance).Where(p => p.Name != "Guard");
            Mods = props.Where(p => p.HasAttr<ModAttribute>()).ToOrderedDictionary(p => p, p => get_value(ptxop, p));
            Affixes = props.Where(p => p.HasAttr<AffixAttribute>()).ToOrderedDictionary(p => p, p => get_value(ptxop, p));
            Operands = props.Where(p => typeof(Expression).IsAssignableFrom(p.PropertyType)).ToOrderedDictionary(p => p, p => get_value(ptxop, p).AssertCast<Expression>());
            var destination = Operands.SingleOrDefault(kvp => kvp.Key.HasAttr<DestinationAttribute>());
            if (destination.Key != null) Destination = Tuple.Create(destination.Key, destination.Value);
        }
    }
}