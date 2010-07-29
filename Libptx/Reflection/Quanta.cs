using System;
using System.Diagnostics;
using System.Linq;
using Libptx.Common.Annotations.Quanta;
using XenoGears;
using XenoGears.Collections.Dictionaries;
using XenoGears.Functional;
using XenoGears.Reflection.Attributes;
using XenoGears.Reflection.Shortcuts;

namespace Libptx.Reflection
{
    [DebuggerNonUserCode]
    public static class QuantumReflector
    {
        public static OrderedDictionary<String, Object> Quanta(this Object obj)
        {
            if (obj == null) return null;
            var schema = new OrderedDictionary<String, Object>();

            var props = obj.GetType().GetProperties(BF.PublicInstance).Where(p => p.HasAttr<QuantumAttribute>()).ToReadOnly();
            props.ForEach((p, i) =>
            {
                var v = p.GetValue(obj, null);
                var @default = p.PropertyType.Fluent(t => t.IsValueType ? Activator.CreateInstance(t) : null);
                if (!Equals(v, @default)) schema.Add(p.Attr<QuantumAttribute>().Signature, v);
            });

            return schema;
        }
    }
}
