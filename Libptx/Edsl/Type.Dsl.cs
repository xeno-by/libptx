using System;
using XenoGears.Functional;
using XenoGears.Assertions;
using System.Linq;

namespace Libptx.Common.Types
{
    public partial class Type
    {
        public Type v1 { get { return Clone(t => t.Mod = TypeMod.V1 | (t.Mod & TypeMod.Array)); } }
        public Type v2 { get { return Clone(t => t.Mod = TypeMod.V2 | (t.Mod & TypeMod.Array)); } }
        public Type v4 { get { return Clone(t => t.Mod = TypeMod.V4 | (t.Mod & TypeMod.Array)); } }
        public Type this[params int[] dims] { get { return Clone(t => t.Mod |= TypeMod.Array, 
            t => t.Dims = (t.Dims ?? Seq.Empty<int>()).Concat(dims.AssertNeitherNullNorEmpty().AssertEach(i => i > 0)).ToArray()); } }

        private Type Clone()
        {
            var clone = new Type();
            clone.Name = this.Name;
            clone.Mod = this.Mod;
            clone.Dims = this.Dims;
            return clone;
        }

        private Type Clone(params Action<Type>[] mods)
        {
            var clone = Clone();
            foreach (var mod in mods) mod(clone);
            return clone;
        }
    }
}
