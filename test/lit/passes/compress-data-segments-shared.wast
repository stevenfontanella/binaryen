;; Test that --compress-data-segments reuses the Huffman symbol decoder and
;; the bit position global that --string-consts-to-data created, setting and
;; restoring the position around its own decoding.

;; RUN: wasm-opt %s -all --string-lowering --string-consts-to-data --pass-arg=string-consts-to-data-always@1 --pass-arg=string-consts-to-data-huffman@1 --compress-data-segments --pass-arg=compress-data-segments-always@1 --pass-arg=compress-data-segments-huffman@1 -S -o - | filecheck %s

(module
  (type $i8 (array (mut i8)))
  (type $col (array externref))

  (global $column (ref $col) (array.new_fixed $col 3
    (string.const "foo")
    (string.const "héllo\n\thello foo")
    (string.const "foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo")
  ))

  (func $read-column (export "col") (param $i i32) (result externref)
    (array.get $col (global.get $column) (local.get $i))
  )

  (data $bytes "Hello, World! abcabcabcabcabcabcabcabcabcabcabcabc")

  (func $bytes (param $off i32) (param $n i32) (result (ref $i8))
    (array.new_data $i8 $bytes (local.get $off) (local.get $n))
  )
)

;; One bits global and one symbol decoder exist, and the segments decoder saves
;; the position, sets its own first bit and restores the position afterwards.
;; (With compress-data-segments-always the strings segment is compressed again
;; as well, which still works: the strings decoder's array.new_data becomes a
;; slice call that decodes the segments first.)
;; CHECK:      (global $lz$bits (mut i32) (i32.const {{[0-9]+}}))
;; CHECK:      (data $segments
;; CHECK-NOT:  (global $lz$bits_
;; CHECK:      (func $lz$symbol (type ${{[0-9]+}}) (param $0 (ref $i8)) (param $1 i32) (result i32)
;; CHECK-NOT:  (func $lz$symbol_
;; CHECK:      (func $segments$decode
;; CHECK:        (local.set $7
;; CHECK-NEXT:    (global.get $lz$bits)
;; CHECK-NEXT:   )
;; CHECK-NEXT:   (global.set $lz$bits
;; CHECK-NEXT:    (i32.const {{[0-9]+}})
;; CHECK-NEXT:   )
;; CHECK:        (data.drop $segments)
;; CHECK:        (global.set $lz$bits
;; CHECK-NEXT:    (local.get $7)
;; CHECK-NEXT:   )
;; CHECK-NEXT:  )
