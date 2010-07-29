using System;
using System.Diagnostics;
using System.Linq;
using System.Collections.Generic;
using Libptx.Expressions.Slots;
using XenoGears.Assertions;
using XenoGears.Collections.Lists;
using XenoGears.Functional;

namespace Libptx
{
    [DebuggerNonUserCode]
    public class Params : BaseList<Var>
    {
        private readonly List<Var> _impl = new List<Var>();
        protected override IEnumerable<Var> Read() { return _impl; }

        public override bool IsReadOnly { get { return false; } }
        protected override void InsertAt(int index, Var el) { _impl.Insert(index, el); }
        protected override void UpdateAt(int index, Var el) { _impl[index] = el; }
        public override void RemoveAt(int index) { _impl.RemoveAt(index); }
    }

    [DebuggerNonUserCode]
    public static class ParamsExtensions
    {
        public static void SetNames(this Params @params, params String[] names)
        {
            SetNames(@params, (IEnumerable<String>)names);
        }

        public static void SetNames(this Params @params, IEnumerable<String> names)
        {
            @params.AssertNotNull();
            names.AssertNotNull();
            (@params.Count() == names.Count()).AssertTrue();
            @params.Zip(names, (p, name) => p.Name = name);
        }
    }
}