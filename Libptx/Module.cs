using System;
using System.Collections.Generic;
using System.IO;
using Libcuda.Versions;
using Libptx.Common;
using Libptx.Expressions;
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

        private Entries _entries = new Entries();
        public Entries Entries
        {
            get { return _entries; }
            set { _entries = value ?? new Entries(); }
        }

        public Entry AddEntry(params Var[] @params)
        {
            return AddEntry((IEnumerable<Var>)@params);
        }

        public Entry AddEntry(String name, params Var[] @params)
        {
            return AddEntry(name, (IEnumerable<Var>)@params);
        }

        public Entry AddEntry(IEnumerable<Var> @params)
        {
            return AddEntry(null, @params);
        }

        public Entry AddEntry(String name, IEnumerable<Var> @params)
        {
            var entry = new Entry();
            entry.Name = name;
            entry.Params.AddElements(@params ?? Seq.Empty<Var>());
            return entry;
        }

        public Module()
            : this(SoftwareIsa.PTX_21, HardwareIsa.SM_10)
        {
        }

        public Module(SoftwareIsa softwareIsa)
            : this(softwareIsa, HardwareIsa.SM_10)
        {
        }

        public Module(HardwareIsa hardwareIsa)
            : this(hardwareIsa < HardwareIsa.SM_20 ? SoftwareIsa.PTX_10 : SoftwareIsa.PTX_20, hardwareIsa)
        {
        }

        public Module(SoftwareIsa softwareIsa, HardwareIsa hardwareIsa)
        {
            Version = softwareIsa;
            Target = hardwareIsa;

            UnifiedTexturing = true;
            EmulateDoubles = false;
            Tuning = new Tuning();
        }

        public Module(HardwareIsa hardwareIsa, SoftwareIsa softwareIsa)
            : this(softwareIsa, hardwareIsa)
        {
        }

        void Validatable.Validate(Module ctx) { (ctx == this).AssertTrue(); Validate(); }
        public void Validate()
        {
            // todo. 128 textures uni mode, 16 samplers non-uni
            // todo. what's the max amount of surfaces?

            (UnifiedTexturing == true).AssertImplies(Version >= SoftwareIsa.PTX_15);
            (EmulateDoubles == true).AssertImplies(Target < HardwareIsa.SM_13);

            Tuning.Validate(this);

            Entries.ForEach(e => e.Validate(this));
        }

        void Renderable.RenderAsPtx(TextWriter writer)
        {
            throw new NotImplementedException();
        }
    }
}
