using System;

namespace Libptx.Expressions
{
    public class Const : Expression
    {
        public static implicit operator Const(bool value)
        {
            throw new NotImplementedException();
        }

        public static implicit operator Const(int value)
        {
            throw new NotImplementedException();
        }

        public static implicit operator Const(uint value)
        {
            throw new NotImplementedException();
        }

        public static implicit operator Const(long value)
        {
            throw new NotImplementedException();
        }

        public static implicit operator Const(ulong value)
        {
            throw new NotImplementedException();
        }

        public static implicit operator Const(float value)
        {
            throw new NotImplementedException();
        }

        public static implicit operator Const(double value)
        {
            throw new NotImplementedException();
        }

        public static implicit operator Const(Array value)
        {
            throw new NotImplementedException();
        }
    }
}