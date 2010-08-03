using System;
using System.Diagnostics;

namespace Libptx.Instructions.Annotations
{
    [AttributeUsage(AttributeTargets.Property, AllowMultiple = false, Inherited = true)]
    [DebuggerNonUserCode]
    public class DestinationAttribute : Attribute
    {
    }
}