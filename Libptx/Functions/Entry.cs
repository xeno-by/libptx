using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Libcuda.Versions;
using Libptx.Common;
using Libptx.Common.Comments;
using Libptx.Common.Performance;
using Libptx.Expressions;
using Libptx.Expressions.Slots;
using Libptx.Statements;
using XenoGears.Assertions;
using XenoGears.Functional;
using Libptx.Common.Names;

namespace Libptx.Functions
{
    [DebuggerNonUserCode]
    public class Entry : Atom
    {
        private String _name = null;
        public virtual String Name
        {
            get { if (_name == null) _name = this.GenName(); return _name; }
            set { _name = value; }
        }

        private Tuning _tuning = new Tuning();
        public virtual Tuning Tuning
        {
            get { return _tuning; }
            set { _tuning = value ?? new Tuning(); }
        }

        private Params _params = new Params();
        public virtual Params Params
        {
            get { return _params; }
            set { _params = value ?? new Params(); }
        }

        private IList<Statement> _stmts = new List<Statement>();
        public virtual IList<Statement> Stmts
        {
            get { return _stmts; }
            set { _stmts = value ?? new List<Statement>(); }
        }

        protected override void CustomValidate()
        {
            Name.ValidateName();
            // uniqueness of names is validated by the context

            Tuning.Validate();

            var size_limit = 256;
            if (ctx.Version >= SoftwareIsa.PTX_15) size_limit += 4096;
            // opaque types don't count against parameter list size limit
            (Params.Sum(p => p.SizeInMemory()) <= size_limit).AssertTrue();

            Params.ForEach(p =>
            {
                p.AssertNotNull();
                p.Validate();
                (p.Space == param).AssertTrue();
            });

            Stmts.ForEach(stmt =>
            {
                stmt.AssertNotNull();
                stmt.Validate();
            });

            var distinct_names = ctx.Visited.Select(a => a is Slot ? ((Slot)a).Name : a is Label ? ((Label)a).Name : null).Where(n => n != null).Distinct().Count();
            var atoms = ctx.Visited.Select(a => a is Slot ? a : a is Label ? a : null).Where(a => a != null).Count();
            (distinct_names == atoms).AssertTrue();

            var declared_labels = ctx.VisitedStmts.OfType<Label>().ToHashSet();
            var referenced_labels = ctx.VisitedExprs.OfType<Label>().ToHashSet();
            declared_labels.IsSupersetOf(referenced_labels).AssertTrue();
        }

        protected override void RenderPtx()
        {
            writer.Write(".entry {0}", Name);

            var nontrivial_tuning = Tuning.Version > SoftwareIsa.PTX_10;
            var nonempty_pragmas = Pragmas.IsNotEmpty();
            if (Params.IsEmpty())
            {
                if (nontrivial_tuning) writer.WriteLine();
                Tuning.RenderPtx();

                if (nonempty_pragmas) writer.WriteLine();
                Pragmas.ForEach(pragma => pragma.RenderPtx());
            }
            else
            {
                // todo. now I've got no idea how to specify pragmas/tuning directives when params ain't empty
                if (nontrivial_tuning || nonempty_pragmas) throw AssertionHelper.Fail();
            }

            // For PTX ISA version 1.4 and later, parameter variables are declared in the kernel
            // parameter list. For PTX ISA versions 1.0 through 1.3, parameter variables are
            // declared in the kernel body.
            if (ctx.Version >= SoftwareIsa.PTX_14)
            {
                if (Params.IsEmpty()) writer.WriteLine("()");
                else
                {
                    writer.WriteLine(" (");
                    indented.Indent++;

                    Params.ForEach((p, i) =>
                    {
                        if (i != 0) writer.WriteLine(",");
                        p.RenderPtx();
                    });

                    indented.Indent--;
                    writer.WriteLine(")");
                }
            }
            else
            {
                writer.WriteLine();
            }

            writer.Delay(() =>
            {
                String curr_prefix = null; int curr_cnt = 0, curr_max = -1;
                Action flush_curr = () =>
                {
                    if (curr_cnt > 1) writer.WriteLine("{0}<{1}>;", curr_prefix, curr_max + 1);
                    else writer.WriteLine("{0};", curr_prefix + (curr_max == -1 ? "" : curr_max.ToString()));
                    curr_prefix = null; curr_cnt = 0; curr_max = -1;
                };

                var regs = ctx.VisitedExprs.OfType<Reg>().OrderBy(reg => reg.Name).ToReadOnly();
                regs.ForEach(reg =>
                {
                    var decl = reg.RunRenderPtx();
                    var m = decl.AssertParse(@"^(?<prefix>.*?)(?<index>\d*)$");
                    var prefix = m["prefix"];
                    var index = m["index"].IsEmpty() ? -1 : int.Parse(m["index"]);
                    if (m["index"].StartsWith("0") && m["index"] != "0") index = -1;

                    if (prefix == curr_prefix && index != -1)
                    {
                        curr_cnt++;
                        curr_max = Math.Max(curr_max, index);
                    }
                    else
                    {
                        if (curr_max != -1) flush_curr();
                        curr_prefix = prefix;
                        curr_cnt = 1;
                        curr_max = index;
                    }
                });
                flush_curr();

                var vars = ctx.VisitedExprs.OfType<Var>().OrderBy(@var => @var.Name).ToReadOnly();
                vars.ForEach(@var =>
                {
                    var opaque = @var.is_opaque();
                    var new_style_param = @var.Space == param && ctx.Version >= SoftwareIsa.PTX_14;
                    if (opaque || new_style_param) return;

                    @var.RenderPtx();
                    writer.WriteLine(";");
                });
            });

            writer.WriteLine("{");
            indented.Indent++;

            foreach (var stmt in Stmts)
            {
                if (stmt is Label)
                {
                    indented.Indent--;
                    stmt.RenderPtx();
                    writer.WriteLine(":");
                    indented.Indent++;
                }
                else if (stmt is Instruction)
                {
                    stmt.RenderPtx();
                    writer.WriteLine(";");
                }
                else if (stmt is Comment)
                {
                    stmt.RenderPtx();
                    writer.WriteLine();
                }
                else
                {
                    throw AssertionHelper.Fail();
                }
            }

            indented.Indent--;
            writer.WriteLine("}");
        }

        public override String ToString()
        {
            return String.Format(".entry {0} ({1})",
                Name.IsNullOrEmpty() ? "<auto>" : Name,
                Params.Select(p => p.RunRenderPtx()).StringJoin());
        }
    }
}