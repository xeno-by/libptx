using System;
using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common;
using Libptx.Common.Names;
using Libptx.Statements;
using Type=Libptx.Common.Types.Type;
using XenoGears.Assertions;
using XenoGears.Strings;

namespace Libptx.Expressions.Slots
{
    [DebuggerNonUserCode]
    public partial class Reg : Atom, Slot, Expression
    {
        private String _name;
        public String Name
        {
            get { if (_name == null) _name = this.GenName(); return _name; }
            set { _name = value; }
        }

        public Type Type { get; set; }

        private int _alignment;
        public int Alignment
        {
            get
            {
                if (_alignment == 0)
                {
                    if (this.is_pred()) return 1;
                    if (this.is_opaque()) return 16;
                    return this.SizeOfElement();
                }
                else
                {
                    return _alignment;
                }
            }

            set { _alignment = value; }
        }

        protected override SoftwareIsa CustomVersion { get { return Type.Version; } }
        protected override HardwareIsa CustomTarget { get { return Type.Target; } }

        protected override void CustomValidate()
        {
            Name.ValidateName();
            // uniqueness of names is validated by the context

            (Type != null).AssertTrue();
            Type.Validate();
            this.is_opaque().AssertFalse();
            this.is_ptr().AssertFalse();
            this.is_bmk().AssertFalse();

            if (_alignment != 0) Alignment.ValidateAlignment(Type);
        }

        protected override void RenderPtx()
        {
            // this is a hack, tho a convenient one
            var render_name = ctx.Parent is Expression || ctx.Parent is Instruction;
            if (render_name)
            {
                writer.Write(Name);
            }
            else
            {
                writer.Write(".reg ");
                if (_alignment != 0) writer.Write(".align " + Alignment + " ");

                var t = Type.PeekRenderPtx();
                var el = t.IndexOf("[") == -1 ? t : t.Slice(0, t.IndexOf("["));
                var indices = t.IndexOf("[") == -1 ? null : t.Slice(t.IndexOf("["));
                writer.Write("{0} {1}{2}", el, Name, indices);
            }
        }

        protected override void RenderCubin()
        {
            throw new NotImplementedException();
        }
    }
}