using System;
using Libcuda.Versions;
using Libptx.Common.Infrastructure;
using Libptx.Expressions;
using Libptx.Specials.Annotations;
using XenoGears.Assertions;

namespace Libptx.Specials
{
    [Special10("%pm{index}", typeof(uint), SoftwareIsa.PTX_13)]
    public class Pm : Special
    {
        [Infix("index")] public int Index { get; set; }

        public override void Validate()
        {
            (0 <= Index && Index <= 3).AssertTrue();
        }
    }
}