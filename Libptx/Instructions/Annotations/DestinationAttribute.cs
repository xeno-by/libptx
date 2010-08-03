using System;

namespace Libptx.Instructions.Annotations
{
    [AttributeUsage(AttributeTargets.Property, AllowMultiple = false, Inherited = true)]
    public class DestinationAttribute : Attribute
    {
    }
}