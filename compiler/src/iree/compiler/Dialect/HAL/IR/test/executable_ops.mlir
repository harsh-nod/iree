// RUN: iree-opt --split-input-file %s | FileCheck %s

#executable_target_format = #hal.executable.target<"backend", "format">
// CHECK-LABEL: @ex
hal.executable @ex {
  // CHECK: hal.executable.variant public @backend, target = #executable_target_format
  hal.executable.variant @backend, target = #executable_target_format {
    // CHECK-DAG: hal.executable.export public @entry0 ordinal(0) layout(#executable_layout) attributes {
    // CHECK-SAME:     workgroup_size = [4 : index, 1 : index, 1 : index]
    hal.executable.export @entry0 ordinal(0) layout(#hal.executable.layout<push_constants = 0, sets = [
      #hal.descriptor_set.layout<0, bindings = [
        #hal.descriptor_set.binding<0, storage_buffer>,
        #hal.descriptor_set.binding<1, storage_buffer>
      ]>
    ]>) attributes {
      workgroup_size = [4 : index, 1 : index, 1 : index]
    }
  }
  // CHECK: hal.executable.binary
  hal.executable.binary @backend_binary attributes {
    // CHECK-SAME: data = dense<1> : vector<128xi8>,
    data = dense<1> : vector<128xi8>,
    // CHECK-SAME: format = "some_format"
    format = "some_format"
  }
}

// -----

#executable_target_format = #hal.executable.target<"backend", "format">

// CHECK-LABEL: @ex_with_workgroup_count_region
hal.executable @ex_with_workgroup_count_region {
  // CHECK: hal.executable.variant public @backend, target = #executable_target_format
  hal.executable.variant @backend, target = #executable_target_format {
    // CHECK-DAG: hal.executable.export public @entry0 ordinal(0) layout(#executable_layout) attributes {
    // CHECK-SAME:     workgroup_size = [4 : index, 1 : index, 1 : index]
    hal.executable.export @entry0 ordinal(0) layout(#hal.executable.layout<push_constants = 0, sets = [
      #hal.descriptor_set.layout<0, bindings = [
        #hal.descriptor_set.binding<0, storage_buffer>,
        #hal.descriptor_set.binding<1, storage_buffer>
      ]>
    ]>) attributes {
      workgroup_size = [4 : index, 1 : index, 1 : index]
    } {
    ^bb0(%device: !hal.device, %arg0: index, %arg1: index, %arg2: index):
      hal.return %arg0, %arg1, %arg2 : index, index, index
    }
  }
  // CHECK: hal.executable.binary
  hal.executable.binary @backend_binary attributes {
    // CHECK-SAME: data = dense<1> : vector<128xi8>,
    data = dense<1> : vector<128xi8>,
    // CHECK-SAME: format = "some_format"
    format = "some_format"
  }
}

// -----

// CHECK-LABEL: @executable_create
// CHECK-SAME: %[[DEVICE:.+]]: !hal.device,
// CHECK-SAME: %[[LAYOUT0:.+]]: !hal.executable_layout,
// CHECK-SAME: %[[LAYOUT1:.+]]: !hal.executable_layout
func.func @executable_create(%device: !hal.device,
                        %layout0: !hal.executable_layout,
                        %layout1: !hal.executable_layout) {
  //      CHECK: = hal.executable.create
  // CHECK-SAME:     device(%[[DEVICE]] : !hal.device)
  // CHECK-SAME:     target(@exe::@binary1)
  // CHECK-SAME:    layouts([%[[LAYOUT0]], %[[LAYOUT1]]]) : !hal.executable
  %0 = hal.executable.create device(%device : !hal.device)
                             target(@exe::@binary1)
                            layouts([%layout0, %layout1]) : !hal.executable
  return
}

// -----

// CHECK-LABEL: @executable_layout_create
// CHECK-SAME: %[[DEVICE:.+]]: !hal.device,
// CHECK-SAME: %[[LAYOUT0:.+]]: !hal.descriptor_set_layout,
// CHECK-SAME: %[[LAYOUT1:.+]]: !hal.descriptor_set_layout
func.func @executable_layout_create(%device: !hal.device,
                               %layout0: !hal.descriptor_set_layout,
                               %layout1: !hal.descriptor_set_layout) {
  // CHECK: hal.executable_layout.create
  // CHECK-SAME:          device(%[[DEVICE]] : !hal.device)
  // CHECK-SAME:  push_constants(1)
  // CHECK-SAME:         layouts([%[[LAYOUT0]], %[[LAYOUT1]]]) : !hal.executable_layout
  %0 = hal.executable_layout.create device(%device : !hal.device)
                            push_constants(1)
                                   layouts([%layout0, %layout1]) : !hal.executable_layout
  return
}

// -----

// CHECK-LABEL: @unresolved_workload_ex
hal.executable @unresolved_workload_ex {
  // CHECK: hal.executable.variant public @backend
  hal.executable.variant @backend, target = #hal.executable.target<"backend", "format"> {
    // CHECK: hal.executable.export public @entry0
    hal.executable.export public @entry0 ordinal(0) layout(#hal.executable.layout<push_constants = 0, sets = [
      #hal.descriptor_set.layout<0, bindings = [
        #hal.descriptor_set.binding<0, storage_buffer>,
        #hal.descriptor_set.binding<1, storage_buffer>
      ]>
    ]>) {
    ^bb0(%device: !hal.device, %arg0: index):
      hal.return %arg0, %arg0, %arg0 : index, index, index
    }
  }
}
// CHECK-LABEL: @unresolved_workload
// CHECK-SAME: (%[[DEVICE:.+]]: !hal.device,
// CHECK-SAME:  %[[WORKLOAD_0:.+]]: index, %[[WORKLOAD_1:.+]]: index)
func.func @unresolved_workload(%device: !hal.device,
                               %workload_0: index, %workload_1: index) -> (index, index, index) {
  // CHECK: %[[WORKGROUP_X:.+]], %[[WORKGROUP_Y:.+]], %[[WORKGROUP_Z:.+]] =
  // CHECK-SAME:   hal.executable.calculate_workgroups
  // CHECK-SAME:       device(%[[DEVICE]] : !hal.device)
  // CHECK-SAME:       target(@unresolved_workload_ex::@backend::@entry0)
  // CHECK-SAME:       workload([%[[WORKLOAD_0]], %[[WORKLOAD_1]]]) : index, index, index
  %workgroups:3 = hal.executable.calculate_workgroups
      device(%device : !hal.device)
      target(@unresolved_workload_ex::@backend::@entry0)
      workload([%workload_0, %workload_1]) : index, index, index
  // CHECK: return %[[WORKGROUP_X]], %[[WORKGROUP_Y]], %[[WORKGROUP_Z]]
  return %workgroups#0, %workgroups#1, %workgroups#2 : index, index, index
}
