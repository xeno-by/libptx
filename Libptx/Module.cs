using System;
using System.Collections.Generic;
using Libcuda.Versions;
using Libptx.Common.Infrastructure;
using XenoGears.Assertions;

namespace Libptx
{
    public class Module : Atom
    {
        public new SoftwareIsa Version { get; set; }
        public new HardwareIsa Target { get; set; }

        public bool UnifiedTexturing { get; set; }
        public bool EmulateDoubles { get; set; }
        public Tuning Tuning { get; set; }

        public IList<Entry> Entries { get; private set; }
        public IList<Func> Funcs { get; private set; }

        public Module()
        {
            Version = SoftwareIsa.PTX_21;
            Target = HardwareIsa.SM_10;

            UnifiedTexturing = true;
            EmulateDoubles = false;
            Tuning = new Tuning();

            Entries = new List<Entry>();
            Funcs = new List<Func>();
        }

        public override void Validate()
        {
            (UnifiedTexturing == true).AssertImplies(Ctx.Version >= SoftwareIsa.PTX_15);
            (EmulateDoubles == true).AssertImplies(Ctx.Target < HardwareIsa.SM_13);

            Tuning.Validate();
        }
    }
}
