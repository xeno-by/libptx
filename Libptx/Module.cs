using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Libcuda.Versions;
using Libptx.Common;
using Libptx.Common.Comments;
using Libptx.Common.Performance;
using Libptx.Common.Performance.Pragmas;
using Libptx.Expressions;
using Libptx.Expressions.Slots;
using Libptx.Instructions;
using XenoGears.Assertions;
using XenoGears.Functional;

namespace Libptx
{
    [DebuggerNonUserCode]
    public class Module : Validatable, Renderable
    {
        public SoftwareIsa Version { get; set; }
        public HardwareIsa Target { get; set; }

        public bool UnifiedTexturing { get; set; }
        public bool EmulateDoubles { get; set; }

        private IList<Pragma> _pragmas = new List<Pragma>();
        public IList<Pragma> Pragmas
        {
            get { return _pragmas; }
            set { _pragmas = value ?? new List<Pragma>(); }
        }

        private IList<Comment> _comments = new List<Comment>();
        public IList<Comment> Comments
        {
            get { return _comments; }
            set { _comments = value ?? new List<Comment>(); }
        }

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
            // this is commented out because there's no problem with UnifiedTexturing
            // if the version is prior to PTX_15, corresponding directive just won't be rendered
//            (UnifiedTexturing == true).AssertImplies(Version >= SoftwareIsa.PTX_15);
            (UnifiedTexturing == false).AssertImplies(Version >= SoftwareIsa.PTX_15);
            (EmulateDoubles == true).AssertImplies(Target < HardwareIsa.SM_13);

            Tuning.Validate(this);
            Pragmas.ForEach(p => { p.AssertNotNull(); p.Validate(this); });
            Comments.ForEach(c => { c.AssertNotNull(); c.Validate(this); });

            var all_args = new HashSet<Expression>();
            Entries.ForEach(e =>
            {
                e.AssertNotNull();
                e.Validate(this);

                (e.Name != null).AssertTrue();
                (Entries.Count(e2 => e2.Name == e.Name) == 1).AssertTrue();

                var args = e.Stmts.OfType<ptxop>().SelectMany(op => op.Operands);
                args.ForEach(arg => all_args.Add(arg));
            });

            if (Target < HardwareIsa.SM_13 && !EmulateDoubles) all_args.AssertNone(arg => arg.is_float() && arg.bits() == 64);
            if (Version < SoftwareIsa.PTX_15) all_args.AssertNone(arg => arg.is_samplerref() || arg.is_surfref());
            (all_args.Where(arg => arg.is_texref()).Count() <= 128).AssertTrue();
            (all_args.Where(arg => arg.is_samplerref()).Count() <= (UnifiedTexturing ? 128 : 16)).AssertTrue();
            // todo. what's the max amount of surfaces?
        }

        void Renderable.RenderAsPtx(TextWriter writer)
        {
            // todo. gather all locations and render their files with ".file" directives

            throw new NotImplementedException();
        }
    }
}
