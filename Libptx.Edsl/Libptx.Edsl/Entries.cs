using System.Collections.Generic;
using XenoGears.Collections;
using XenoGears.Functional;
using System.Linq;

namespace Libptx.Edsl
{
    public static partial class Ptx21
    {
        public static partial class Sm20
        {
            public class Entries : BaseList<Entry>
            {
                public Entries(params Entry[] entries) : this((IEnumerable<Entry>)entries) {}
                public Entries(IEnumerable<Entry> entries) : this(new Libptx.Entries((entries ?? Seq.Empty<Entry>()).Select(e => (Libptx.Entry)e))) {}

                private readonly Libptx.Entries _base;
                internal Entries(Libptx.Entries @base) { _base = @base; }
                public static implicit operator Libptx.Entries(Entries entries) { return entries == null ? null : entries._base; }

                protected override IEnumerable<Entry> Read() { return _base.Select(e => e is Entry ? (Entry)e : new Entry(e)); }
                public override bool IsReadOnly { get { return _base.IsReadOnly; } }
                protected override void InsertAt(int index, Entry el) { _base.Insert(index, el); }
                protected override void UpdateAt(int index, Entry el) { _base[index] = el; }
                public override void RemoveAt(int index) { _base.RemoveAt(index); }
            }
        }
    }
}