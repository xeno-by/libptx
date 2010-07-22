using System;
using System.Collections.Generic;
using Libcuda.Versions;
using Libptx.Expressions;
using XenoGears.Functional;

namespace Libptx.Edsl
{
    public static partial class Ptx21
    {
        public static partial class Sm20
        {
            public static Module NewModule()
            {
                return new Module();
            }

            public class Module : Libptx.Module
            {
                public new Entries Entries
                {
                    get { return new Entries(base.Entries); }
                    set { base.Entries = value ?? new Entries(); }
                }

                public new Entry AddEntry(params Var[] @params)
                {
                    return AddEntry((IEnumerable<Var>)@params);
                }

                public new Entry AddEntry(String name, params Var[] @params)
                {
                    return AddEntry(name, (IEnumerable<Var>)@params);
                }

                public new Entry AddEntry(IEnumerable<Var> @params)
                {
                    return AddEntry(null, @params);
                }

                public new Entry AddEntry(String name, IEnumerable<Var> @params)
                {
                    var entry = new Entry();
                    entry.Name = name;
                    entry.Params.AddElements(@params ?? Seq.Empty<Var>());
                    return entry;
                }

                public Module()
                    : base(SoftwareIsa.PTX_21, HardwareIsa.SM_20)
                {
                }
            }
        }
    }
}
