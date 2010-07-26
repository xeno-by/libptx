using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using Libcuda.DataTypes;
using Libcuda.Versions;
using Libptx.Common;
using XenoGears.Assertions;
using System.Linq;

namespace Libptx
{
    [DebuggerNonUserCode]
    public class Tuning : Atom
    {
        public int Maxnreg { get; set; }
        public dim3 Maxntid { get; set; }
        public dim3 Reqntid { get; set; }
        public int Minnctapersm { get; set; }
        public int Maxnctapersm { get; set; }

        protected override SoftwareIsa CustomVersion
        {
            get
            {
                var mods = new List<SoftwareIsa>();
                if (Maxnreg > 0) mods.Add(SoftwareIsa.PTX_13);
                if (Maxntid != null) mods.Add(SoftwareIsa.PTX_13);
                if (Reqntid != null) mods.Add(SoftwareIsa.PTX_21);
                if (Minnctapersm != 0) mods.Add(SoftwareIsa.PTX_20);
                if (Maxnctapersm != 0) mods.Add(SoftwareIsa.PTX_13);
                return mods.Max();
            }
        }

        protected override void CustomValidate(Module ctx)
        {
            (Maxnreg >= 0).AssertTrue();
            (Minnctapersm >= 0).AssertTrue();
            (Maxnctapersm >= 0).AssertTrue();
        }

        protected override void RenderAsPtx(TextWriter writer)
        {
            throw new NotImplementedException();
        }
    }
}