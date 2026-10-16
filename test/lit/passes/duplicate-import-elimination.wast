;; RUN: foreach %s %t wasm-opt --duplicate-import-elimination -S -o - | filecheck %s

(module
  (type $sig1 (func (param i32) (result i32)))
  (type $sig2 (func (param i64) (result i64)))

  ;; Interleaved overloaded imports with the same (module, base) but different
  ;; signatures. $get1 and $get3 share $sig1 and should be deduplicated even
  ;; though $get2 with $sig2 appears between them.
  ;; CHECK:      (import "imports" "getAt" (func $get1 (param i32) (result i32)))
  (import "imports" "getAt" (func $get1 (type $sig1) (param i32) (result i32)))
  ;; CHECK:      (import "imports" "getAt" (func $get2 (param i64) (result i64)))
  (import "imports" "getAt" (func $get2 (type $sig2) (param i64) (result i64)))
  ;; CHECK-NOT:  (import "imports" "getAt" (func $get3
  (import "imports" "getAt" (func $get3 (type $sig1) (param i32) (result i32)))

  ;; CHECK:      (func $caller (param $x i32) (param $y i64) (result i32)
  ;; CHECK-NEXT:  (drop
  ;; CHECK-NEXT:   (call $get2
  ;; CHECK-NEXT:    (local.get $y)
  ;; CHECK-NEXT:   )
  ;; CHECK-NEXT:  )
  ;; CHECK-NEXT:  (i32.add
  ;; CHECK-NEXT:   (call $get1
  ;; CHECK-NEXT:    (local.get $x)
  ;; CHECK-NEXT:   )
  ;; CHECK-NEXT:   (call $get1
  ;; CHECK-NEXT:    (local.get $x)
  ;; CHECK-NEXT:   )
  ;; CHECK-NEXT:  )
  ;; CHECK-NEXT: )
  (func $caller (export "caller") (param $x i32) (param $y i64) (result i32)
    (drop
      (call $get2
        (local.get $y)
      )
    )
    (i32.add
      (call $get1
        (local.get $x)
      )
      (call $get3
        (local.get $x)
      )
    )
  )
)
