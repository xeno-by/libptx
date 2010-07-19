using System;
using XenoGears.Functional;
using XenoGears.Assertions;
using System.Linq;

namespace Libptx.Common.Types
{
    public partial class Type
    {
        public Type v1 { get { (Mod == TypeMod.Scalar).AssertTrue(); return Clone(t => t.Mod = TypeMod.V1); } }
        public Type v2 { get { (Mod == TypeMod.Scalar).AssertTrue(); return Clone(t => t.Mod = TypeMod.V2); } }
        public Type v4 { get { (Mod == TypeMod.Scalar).AssertTrue(); return Clone(t => t.Mod = TypeMod.V4); } }
        public Type this[params int[] dims] { get { return Clone(t => t.Mod |= TypeMod.Array, 
            t => t.Dims = (t.Dims ?? Seq.Empty<int>()).Concat(dims.AssertNeitherNullNorEmpty().AssertEach(i => i > 0)).ToArray()); } }
        public Type x { get { ((Mod & TypeMod.V1) == TypeMod.V1 || (Mod & TypeMod.V2) == TypeMod.V2 || (Mod & TypeMod.V4) == TypeMod.V4).AssertTrue(); return Clone(t => t.Mod &= TypeMod.Array); } }
        public Type r { get { ((Mod & TypeMod.V1) == TypeMod.V1 || (Mod & TypeMod.V2) == TypeMod.V2 || (Mod & TypeMod.V4) == TypeMod.V4).AssertTrue(); return Clone(t => t.Mod &= TypeMod.Array); } }
        public Type y { get { ((Mod & TypeMod.V2) == TypeMod.V2 || (Mod & TypeMod.V4) == TypeMod.V4).AssertTrue(); return Clone(t => t.Mod &= TypeMod.Array); } }
        public Type g { get { ((Mod & TypeMod.V2) == TypeMod.V2 || (Mod & TypeMod.V4) == TypeMod.V4).AssertTrue(); return Clone(t => t.Mod &= TypeMod.Array); } }
        public Type z { get { ((Mod & TypeMod.V4) == TypeMod.V4).AssertTrue(); return Clone(t => t.Mod &= TypeMod.Array); } }
        public Type b { get { ((Mod & TypeMod.V4) == TypeMod.V4).AssertTrue(); return Clone(t => t.Mod &= TypeMod.Array); } }
        public Type w { get { ((Mod & TypeMod.V4) == TypeMod.V4).AssertTrue(); return Clone(t => t.Mod &= TypeMod.Array); } }
        public Type a { get { ((Mod & TypeMod.V4) == TypeMod.V4).AssertTrue(); return Clone(t => t.Mod &= TypeMod.Array); } }

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
