using System.Collections.Generic;
using System.Diagnostics;
using XenoGears.Collections.Lists;
using XenoGears.Functional;

namespace Libptx
{
    [DebuggerNonUserCode]
    public class Entries : BaseList<Entry>
    {
        public Entries(params Entry[] entries) : this((IEnumerable<Entry>)entries) {}
        public Entries(IEnumerable<Entry> entries) { _impl = new List<Entry>(entries ?? Seq.Empty<Entry>()); }

        private readonly List<Entry> _impl = new List<Entry>();
        protected override IEnumerable<Entry> Read() { return _impl; }

        public override bool IsReadOnly { get { return false; } }
        protected override void InsertAt(int index, Entry el) { _impl.Insert(index, el); }
        protected override void UpdateAt(int index, Entry el) { _impl[index] = el; }
        public override void RemoveAt(int index) { _impl.RemoveAt(index); }
    }
}