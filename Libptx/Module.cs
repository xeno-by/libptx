using System;
using System.Collections.Generic;
using System.IO;
using Libcuda.Versions;
using Libptx.Common;
using XenoGears.Assertions;
using XenoGears.Functional;

namespace Libptx
{
    public class Module : Validatable, Renderable
    {
        public SoftwareIsa Version { get; set; }
        public HardwareIsa Target { get; set; }

        public bool UnifiedTexturing { get; set; }
        public bool EmulateDoubles { get; set; }

        private Tuning _tuning = new Tuning();
        public Tuning Tuning
        {
            get { return _tuning; }
            set { _tuning = value ?? new Tuning(); }
        }

        private IList<Entry> _entries = new List<Entry>();
        public IList<Entry> Entries
        {
            get { return _entries; }
            set { _entries = value ?? new List<Entry>(); }
        }

        private IList<Func> _funcs = new List<Func>();
        public IList<Func> Funcs
        {
            get { return _funcs; }
            set { _funcs = value ?? new List<Func>(); }
        }

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

        void Validatable.Validate(Module ctx) { (ctx == this).AssertTrue(); Validate(); }
        public void Validate()
        {
            (UnifiedTexturing == true).AssertImplies(Version >= SoftwareIsa.PTX_15);
            (EmulateDoubles == true).AssertImplies(Target < HardwareIsa.SM_13);

            Tuning.Validate(this);

            Entries.ForEach(e => e.Validate(this));
            Funcs.ForEach(f => f.Validate(this));
        }

        void Renderable.RenderAsPtx(TextWriter writer)
        {
            throw new NotImplementedException();
        }
    }
}
