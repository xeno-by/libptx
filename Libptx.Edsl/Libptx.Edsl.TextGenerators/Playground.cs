using System;

namespace Libptx.Edsl.TextGenerators
{
    public class pg
    {
        public void foo()
        {
            var_u32 d = null;
            var_u32 a = null;
            var_s32 b = null;
            ptx.add(d, a, b);
            ptx.cvt_s32_u16(d, a);
        }
    }

    public class ptx
    {
        public static void add(u32 d, u32 a, u32 b)
        {

        }

        public static void cvt_s32_u16(relaxed_u16 d, relaxed_s32 a)
        {
        }
    }

    public class type { }
    public class relaxed_type<T> where T : type { }

    public class u32 : type { }
    public class relaxed_u32 : relaxed_type<u32> { }
    public class s32 : type { }
    public class relaxed_s32 : relaxed_type<s32> { }
    public class u16 : type { }
    public class relaxed_u16 : relaxed_type<u16> { }
    public class s16 : type { }
    public class relaxed_s16 : relaxed_type<s16> { }

    public class has_type_u32
    {
        public static implicit operator u32(has_type_u32 u32) { throw new NotImplementedException(); }
        public static implicit operator s32(has_type_u32 u32) { throw new NotImplementedException(); }
        public static implicit operator relaxed_u32(has_type_u32 u32) { throw new NotImplementedException(); }
        public static implicit operator relaxed_s32(has_type_u32 u32) { throw new NotImplementedException(); }
        public static implicit operator relaxed_u16(has_type_u32 u32) { throw new NotImplementedException(); }
        public static implicit operator relaxed_s16(has_type_u32 u32) { throw new NotImplementedException(); }
    }

    public class has_type_s32
    {
        public static implicit operator u32(has_type_s32 s32) { throw new NotImplementedException(); }
        public static implicit operator s32(has_type_s32 s32) { throw new NotImplementedException(); }
        public static implicit operator relaxed_u32(has_type_s32 s32) { throw new NotImplementedException(); }
        public static implicit operator relaxed_s32(has_type_s32 s32) { throw new NotImplementedException(); }
        public static implicit operator relaxed_u16(has_type_s32 s32) { throw new NotImplementedException(); }
        public static implicit operator relaxed_s16(has_type_s32 s32) { throw new NotImplementedException(); }
    }

    public class has_type_u16
    {
        public static implicit operator u16(has_type_u16 u16) { throw new NotImplementedException(); }
        public static implicit operator s16(has_type_u16 u16) { throw new NotImplementedException(); }
        public static implicit operator relaxed_u16(has_type_u16 u16) { throw new NotImplementedException(); }
        public static implicit operator relaxed_s16(has_type_u16 u16) { throw new NotImplementedException(); }
    }

    public class has_type_s16
    {
        public static implicit operator u16(has_type_s16 s16) { throw new NotImplementedException(); }
        public static implicit operator s16(has_type_s16 s16) { throw new NotImplementedException(); }
        public static implicit operator relaxed_u16(has_type_s16 s16) { throw new NotImplementedException(); }
        public static implicit operator relaxed_s16(has_type_s16 s16) { throw new NotImplementedException(); }
    }

    public class var_u32 : has_type_u32
    {
    }

    public class var_s32 : has_type_s32
    {
    }

    public class var_u16 : has_type_u16
    {
    }

    public class var_s16 : has_type_s16
    {
    }

}
