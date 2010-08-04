using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Libcuda.Versions;
using Libptx.Common;
using Libptx.Common.Comments;
using Libptx.Common.Contexts;
using Libptx.Common.Performance.Pragmas;
using Libptx.Common.Types;
using Libptx.Expressions;
using Libptx.Expressions.Slots;
using Libptx.Functions;
using XenoGears.Assertions;
using XenoGears.Functional;
using Type = Libptx.Common.Types.Type;

namespace Libptx
{
    [DebuggerNonUserCode]
    public class Module
    {
        public SoftwareIsa Version { get; set; }
        public HardwareIsa Target { get; set; }

        public bool UnifiedTexturing { get; set; }
        public bool DowngradeDoubles { get; set; }

        private IList<Comment> _comments = new List<Comment>();
        public IList<Comment> Comments
        {
            get { return _comments; }
            set { _comments = value ?? new List<Comment>(); }
        }

        private IList<Pragma> _pragmas = new List<Pragma>();
        public IList<Pragma> Pragmas
        {
            get { return _pragmas; }
            set { _pragmas = value ?? new List<Pragma>(); }
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
            Entries.Add(entry);
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
            DowngradeDoubles = false;
        }

        public Module(HardwareIsa hardwareIsa, SoftwareIsa softwareIsa)
            : this(softwareIsa, hardwareIsa)
        {
        }

        public void Validate()
        {
            var ctx = new ValidationContext(this);
            using (ValidationContext.Push(ctx))
            {
                // this is commented out because there's no problem with UnifiedTexturing
                // if the version is prior to PTX_15, corresponding directive just won't be rendered
//                (UnifiedTexturing == true).AssertImplies(Version >= SoftwareIsa.PTX_15);
                (UnifiedTexturing == false).AssertImplies(Version >= SoftwareIsa.PTX_15);
                (DowngradeDoubles == true).AssertImplies(Target < HardwareIsa.SM_13);

                Comments.ForEach(c => { c.AssertNotNull(); c.Validate(); });
                Pragmas.ForEach(p => { p.AssertNotNull(); p.Validate(); });
                Entries.ForEach(e => { e.AssertNotNull(); e.Validate(); });
                (Entries.Count() == Entries.Select(e => e.Name).Distinct().Count()).AssertTrue();

                Func<TypeName, bool> mentioned_type = t => ctx.Visited.Contains((Type)t);
                if (Target < HardwareIsa.SM_13 && !DowngradeDoubles) mentioned_type(TypeName.F64).AssertFalse();
                if (Version < SoftwareIsa.PTX_15) (mentioned_type(TypeName.Samplerref) || mentioned_type(TypeName.Surfref)).AssertFalse();
                (ctx.VisitedExprs.Where(arg => arg.is_texref()).Count() <= 128).AssertTrue();
                (ctx.VisitedExprs.Where(arg => arg.is_samplerref()).Count() <= (UnifiedTexturing ? 128 : 16)).AssertTrue();
                // todo. what's the max amount of surfaces?
            }
        }

        public String RenderPtx()
        {
            var ctx = new RenderPtxContext(this);
            using (RenderPtxContext.Push(ctx))
            {
                Comments.ForEach(c => c.RenderPtx());
                if (Comments.IsNotEmpty()) ctx.Writer.WriteLine();

                ctx.Writer.WriteLine(".version {0}.{1}", (int)Version / 10, (int)Version % 10);
                ctx.Writer.Write(".target sm_{0}", (int)Target);
//                if (Version >= SoftwareIsa.PTX_15 && UnifiedTexturing == true) ctx.Writer.Write(", texmode_unified");
                if (Version >= SoftwareIsa.PTX_15 && UnifiedTexturing == false) ctx.Writer.Write(", texmode_independent");
                if (DowngradeDoubles) ctx.Writer.Write(", map_f64_to_f32");
                ctx.Writer.WriteLine();

                if (Pragmas.IsNotEmpty()) ctx.Writer.WriteLine();
                Pragmas.ForEach(p => p.RenderPtx());

                ctx.DelayRender(() =>
                {
                    var opaques = ctx.VisitedExprs.OfType<Var>().Where(v => v.is_opaque()).ToReadOnly();
                    if (opaques.IsNotEmpty()) ctx.Writer.WriteLine();
                    opaques.ForEach(v => v.RenderPtx());
                });

                foreach (var entry in Entries)
                {
                    ctx.Writer.WriteLine();
                    entry.RenderPtx();
                }

                ctx.CommitRender();
                return ctx.Result;
            }
        }

        public byte[] RenderCubin()
        {
            throw new NotImplementedException();
        }
    }
}
