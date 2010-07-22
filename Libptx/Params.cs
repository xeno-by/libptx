using System.Collections.Generic;
using Libptx.Expressions;
using XenoGears.Collections;

namespace Libptx
{
    public class Params : BaseList<Var>
    {
        private readonly List<Var> _impl = new List<Var>();
        protected override IEnumerable<Var> Read() { return _impl; }

        protected override bool IsReadOnly { get { return false; } }
        protected override void InsertAt(int index, Var el) { _impl.Insert(index, el); }
        protected override void UpdateAt(int index, Var el) { _impl[index] = el; }
        public override void RemoveAt(int index) { _impl.RemoveAt(index); }
    }
}