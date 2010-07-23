using System;
using Libptx.Edsl.Statements;

namespace Libptx.Edsl
{
    public static partial class Ptx21
    {
        public static partial class Sm20
        {
            public class Entry_Aux : Libptx.Entry
            {
                protected Libptx.Statements.Block GetBody() { return base.Body; }
                protected void SetBody(Libptx.Statements.Block value) { base.Body = value; }

                public sealed override Libptx.Statements.Block Body
                {
                    get { return GetBody(); }
                    set { SetBody(value); }
                }
            }

            public class Entry : Libptx.Entry
            {
                private readonly Libptx.Entry _base;
                public Entry() : this(new Libptx.Entry()) { }
                internal Entry(Libptx.Entry @base) { _base = @base; }

                public override String Name
                {
                    get { return _base.Name; }
                    set { _base.Name = value; }
                }

                public override Tuning Tuning
                {
                    get { return _base.Tuning; }
                    set { _base.Tuning = value; }
                }

                public override Params Params
                {
                    get { return _base.Params; }
                    set { _base.Params = value; }
                }

                public new Block Body
                {
                    get { return _base.Body is Block ? (Block)_base.Body : new Block(_base.Body); }
                    set { _base.Body = value; }
                }
            }
        }
    }
}