using System;
using Libptx.Common.Enumerations;
using XenoGears.Assertions;
using Type=Libptx.Common.Type;

namespace Libptx.Expressions
{
    public abstract class Special : Var
    {
        protected virtual Type CustomType { get { throw new NotImplementedException(); } }

        public override String Name
        {
            get { throw new NotImplementedException(); }
            set { throw AssertionHelper.Fail(); }
        }

        public override Space Space
        {
            get { return Space.Special; }
            set { throw AssertionHelper.Fail(); }
        }

        public override Type Type
        {
            get { throw new NotImplementedException(); }
            set { throw AssertionHelper.Fail(); }
        }

        public override Const Init
        {
            get { return null; }
            set { throw AssertionHelper.Fail(); }
        }

        public override int Alignment
        {
            get { return 0; }
            set { throw AssertionHelper.Fail(); }
        }

        public override bool IsVisible
        {
            get { return false; }
            set { throw AssertionHelper.Fail(); }
        }

        public override bool IsExtern
        {
            get { return false; }
            set { throw AssertionHelper.Fail(); }
        }

        public override VarMod Mod
        {
            get { return 0; }
            set { throw AssertionHelper.Fail(); }
        }
    }
}