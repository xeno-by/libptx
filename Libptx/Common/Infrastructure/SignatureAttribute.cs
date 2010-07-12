using System;

namespace Libptx.Common.Infrastructure
{
    [AttributeUsage(AttributeTargets.Field, AllowMultiple = true, Inherited = true)]
    public class SignatureAttribute : Attribute
    {
        public virtual String Signature { get; set; }

        public SignatureAttribute(String signature)
        {
            Signature = signature;
        }
    }
}