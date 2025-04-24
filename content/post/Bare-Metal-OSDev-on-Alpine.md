---
title: "Bare-Metal OSDev on Alpine"
date: 2025-04-24
id: 9
author: "Afnan K Salal"
authorGithub: "https://github.com/afnanksalal"
tags:
  - OSDev
  - Alpine
  - GRUB
  - Bare-Metal
  - Virtualization
  - Cross-Compilation
  - i386-elf
---

## **Bare-Metal OSDev on Alpine**

> *“The pursuit began innocently enough: to craft an operating system from the ground up. What unfolded was a harrowing descent into the depths of Alpine Linux, GRUB intricacies, and the unforgiving landscape of nested virtualization.”*

### **The Labyrinth of Virtualization: A Multi-Layered Reality**

Most aspiring kernel developers begin with a single, straightforward virtual machine. My approach, however, was somewhat... unconventional. I elected to nest a bare Alpine Linux VM within a Windows host environment and then, inexplicably, proceeded to execute QEMU *inside* the Alpine VM. This multi-layered virtualization strategy, while intellectually stimulating in theory, quickly devolved into a complex debugging exercise. The underlying rationale remains elusive, perhaps a misguided attempt to emulate the operational environment of embedded systems, or simply a desire to inflict maximum complexity.

### **Alpine: A Spartan Landscape and the OSDev Gauntlet**

The initial encounter with the Alpine console served as a stark reminder of its minimalist philosophy. Alpine is not merely lean; it represents a bare-bones, stripped-down operating environment. The intent to compile C code immediately necessitated the acquisition of the GNU Compiler Collection (GCC) via `apk add gcc`, a command that belies the underlying intricacies of Alpine's package management system (`apk`). Similarly, assembling assembly code demanded a comprehensive search for the GNU Binutils suite. GRUB, the Grand Unified Bootloader, presented its own set of challenges. The available installation was a skeletal framework, lacking the essential components required to construct a bootable image.

The subsequent phase involved an exhaustive exploration of Alpine's package ecosystem, a rigorous hunt for build dependencies, and a painstaking examination of the nuances of the `musl` C standard library implementation. The challenges rapidly escalated from basic package installation to grappling with the intricacies of 32-bit compilation (`gcc -m32`) within an increasingly 64-bit dominated landscape. The Alpine Wiki became an indispensable resource, serving as a critical bridge between cryptic error messages and actionable solutions. References to the Alpine Packages Search proved invaluable in identifying correct packages and versions, particularly given the nuances of Alpine's rolling release model.

### **The Cross-Compiler Conundrum: A Toolchain From Scratch**

Alpine's adoption of `musl`, a lightweight C standard library tailored for efficiency and static linking, introduces a significant obstacle when targeting a bare-metal x86 environment that conforms to the GRUB Multiboot Specification. `musl` subtly alters fundamental assumptions pertaining to linking, system calls, and the expected memory layout of executables. The solution, both elegant and arduous, involved the creation of a custom `i386-elf` cross-compiler toolchain from source.

This task transcended a simple `apt-get install build-essential` command. It required the manual retrieval of Binutils source code, configuring it with the `--target=i686-elf` flag, and initiating a lengthy build process. Subsequently, the GCC source code had to be downloaded, configured with `--target=i686-elf --disable-libssp --without-headers`, and subjected to an even more protracted compilation cycle. The initial attempts resulted in a cascade of errors, ranging from unresolved dependencies to misconfigured compilation flags and unresolved linker errors. Subsequent iterations involved meticulous adjustments, further periods of waiting, and the resolution of newly emergent errors. The culminating moment, the successful execution of `i686-elf-gcc` and `i686-elf-ld` without generating a torrent of diagnostic messages, felt akin to a pyrrhic victory. The Newlib C library served as an unsung partner in this endeavor, providing the necessary runtime environment, albeit at the cost of increased complexity. The [GNU toolchain documentation](https://www.gnu.org/software/binutils/) and [GCC documentation](https://gcc.gnu.org/onlinedocs/) were invaluable resources, although their complexity often matched the problem at hand.

### **GRUB: An Expedition into Bootloader Forensics**

The Alpine GRUB installation, while present, lacked the crucial `i386-pc` modules essential for generating a bootable ISO image. The resulting error message, now etched into my memory:

```sh
grub-mkrescue: error: /usr/lib/grub/i386-pc not found
```

This necessitated the compilation of GRUB *itself* from source. Navigating the GRUB build system, deciphering autoconf options, and overcoming dependency conflicts became the new operational paradigm. Extensive hours were dedicated to scrutinizing GRUB documentation, much of which appeared outdated or incomplete, and seeking guidance from online forums, particularly Stack Overflow. The eventual emergence of the `.mod` files within `/usr/local/lib/grub/i386-pc` represented a significant milestone. The `grub-mkrescue` command finally executed without complaint.

### **That Initial 'X': A Pixel of Triumph**

The bootloader code itself was deliberately minimalistic. It was designed to load the kernel image into memory, write the character `'X'` to the upper-left corner of the screen using direct video memory access in VGA mode, and then transfer control to the `kernel_main` function.

Contrary to all expectations, it functioned as intended.

Witnessing the appearance of that solitary, well-defined `'X'` upon booting QEMU, succeeded by the display of the message "Hello from kernel!", induced a feeling of elation. It represented the culmination of countless hours of frustration, rigorous debugging, and unwavering persistence. It was akin to establishing a direct, unmediated line of communication with the machine.

### **Lessons Learned from the Abyss: Retrospection and Regret**

- **Avoid Alpine for OSDev unless thoroughly prepared for a demanding challenge.** While a versatile distribution, it is not ideally suited for novice kernel developers building from scratch.
- **Kernel development is relatively straightforward; the toolchain setup is the primary hurdle.** Kernel panics are manageable; compiler errors are the true adversaries.
- **Nested virtualization is theoretically intriguing, but debugging across multiple layers is problematic.** Consider a simplified VM configuration.
- **Compiling GRUB from source develops character and potentially induces trauma.**
- **The ultimate success is rewarding.** The experience is often worth the hardship.
