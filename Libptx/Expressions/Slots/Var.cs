using System;
using System.Diagnostics;
using System.Linq;
using Libcuda.Versions;
using Libptx.Common;
using Libptx.Common.Enumerations;
using Libptx.Common.Names;
using Libptx.Common.Types;
using Libptx.Common.Types.Pointers;
using Libptx.Expressions.Addresses;
using Libptx.Expressions.Immediate;
using Libptx.Reflection;
using XenoGears.Functional;
using Type = Libptx.Common.Types.Type;
using XenoGears.Assertions;
using XenoGears.Strings;

namespace Libptx.Expressions.Slots
{
    [DebuggerNonUserCode]
    public partial class Var : Atom, Slot, Addressable, Expression
    {
        private String _name;
        public String Name
        {
            get { if (_name == null) _name = this.GenName(); return _name; }
            set { _name = value; }
        }

        public space Space { get; set; }

        // this is a very important point for comprehending Libptx
        //
        // despite of spec implicitly uniting registers and vars into a single concept
        // we have three distinct types of variables: regs, sregs and vars
        // every one of those is very different from all others
        //
        // in fact, vars are just pointers, i.e.:
        // 1) they cannot participate in ALU instructions,
        // 2) when used in ld/st they must be put in brackets (i.e. dereferenced ala *ptr in C-like languages)
        //
        // that's why vars do have the typeof(Ptr) type
        //
        // however, when dealing with opaques we need to treat them differently
        // since opaques are themselves pointers, they just have different flavor
        // n0te that for being used with tex/surf instructions, we need to put opaques in brackets
        // which is, as described above, a direct analogue of pointer dereferencing
        Type Expression.Type { get { return Type.is_opaque() ? Type : (Type)typeof(Ptr); } }
        public Type Type { get; set; }

        public Const Init { get; set; }

        private int _alignment;
        public int Alignment
        {
            get
            {
                if (_alignment == 0)
                {
                    if (this.Type.is_opaque()) return 16;
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

            (Space != 0).AssertTrue();

            (Type != null).AssertTrue();
            Type.Validate();
            this.Type.is_pred().AssertFalse();
            this.Type.is_opaque().AssertImplies(Space == param || Space == global);
            this.Type.is_ptr().AssertFalse();
            this.Type.is_bmk().AssertFalse();

            agree_or_null(Init, Type);
            (Init != null).AssertImplies(Space.is_const() || Space == global);
            (Init != null).AssertImplies(Type != f16 && Type != pred);
            // todo. ".global variables used in initializers, the resulting address is a generic address"
            // wtf does this mean? is this feature limited to SM_20 or that was just unintended pun?
            var init_var = Init == null ? null : Init.Value as Var;
            if (init_var != null) (init_var.Space.is_const() || init_var.Space == global).AssertTrue();

            if (_alignment != 0) Alignment.ValidateAlignment(Type);
        }

        protected override void RenderPtx()
        {
            // this is a hack, tho a convenient one
            var render_name = ctx.Parent is Expression;
            if (render_name)
            {
                writer.Write(Name);
            }
            else
            {
                if (Type.is_texref() && ctx.Version < SoftwareIsa.PTX_15)
                {
                    writer.Write(".tex .u32 {0}", Name);
                }
                else
                {
                    writer.Write("." + Space.Signature() + " ");
                    if (_alignment != 0) writer.Write(".align " + Alignment + " ");

                    var t = Type.PeekRenderPtx();
                    var el = t.IndexOf("[") == -1 ? t : t.Slice(0, t.IndexOf("[") - 1);
                    var indices = t.IndexOf("[") == -1 ? null : t.Slice(t.IndexOf("["));
                    if (Init != null) indices = (Type.Dims ?? new int[0]).Count().Times("[]");

                    writer.Write(".{0} {1}{2}", el, Name, indices);
                    if (Init != null) writer.Write(" = ");
                    if (Init != null) Init.RenderPtx();
                }
            }
        }

        protected override void RenderCubin()
        {
            throw new NotImplementedException();
        }
    }
}
