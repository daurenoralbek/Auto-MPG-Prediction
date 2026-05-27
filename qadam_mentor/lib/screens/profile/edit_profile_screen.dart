import 'dart:io';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:image_picker/image_picker.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../config/app_colors.dart';
import '../../providers/auth_provider.dart';
import '../../services/firestore_service.dart';
import '../../services/storage_service.dart';
import '../../widgets/common/custom_button.dart';
import '../../widgets/common/custom_text_field.dart';

class EditProfileScreen extends StatefulWidget {
  const EditProfileScreen({super.key});

  @override
  State<EditProfileScreen> createState() => _EditProfileScreenState();
}

class _EditProfileScreenState extends State<EditProfileScreen> {
  final _formKey = GlobalKey<FormState>();
  final _nameCtrl = TextEditingController();
  final _bioCtrl = TextEditingController();
  final _specialtyCtrl = TextEditingController();
  final _phoneCtrl = TextEditingController();
  String _selectedYear = '1';
  File? _pickedImage;
  bool _isLoading = false;

  final _firestoreService = FirestoreService();
  final _storageService = StorageService();

  @override
  void initState() {
    super.initState();
    final user = context.read<AuthProvider>().user;
    if (user != null) {
      _nameCtrl.text = user.name;
      _bioCtrl.text = user.bio ?? '';
      _specialtyCtrl.text = user.specialty ?? '';
      _phoneCtrl.text = user.phone ?? '';
      _selectedYear = user.year ?? '1';
    }
  }

  @override
  void dispose() {
    _nameCtrl.dispose();
    _bioCtrl.dispose();
    _specialtyCtrl.dispose();
    _phoneCtrl.dispose();
    super.dispose();
  }

  Future<void> _pickImage() async {
    final picker = ImagePicker();
    final source = await showModalBottomSheet<ImageSource>(
      context: context,
      builder: (_) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ListTile(
              leading: const Icon(Icons.camera_alt_outlined),
              title: Text('Камера', style: GoogleFonts.nunito()),
              onTap: () => Navigator.pop(context, ImageSource.camera),
            ),
            ListTile(
              leading: const Icon(Icons.photo_library_outlined),
              title: Text('Галерея', style: GoogleFonts.nunito()),
              onTap: () => Navigator.pop(context, ImageSource.gallery),
            ),
          ],
        ),
      ),
    );
    if (source == null) return;
    final picked = await picker.pickImage(
        source: source, maxWidth: 800, imageQuality: 85);
    if (picked != null) setState(() => _pickedImage = File(picked.path));
  }

  Future<void> _save() async {
    if (!_formKey.currentState!.validate()) return;
    setState(() => _isLoading = true);

    try {
      final authProv = context.read<AuthProvider>();
      final uid = authProv.user!.uid;
      String? photoUrl = authProv.user?.photoUrl;

      if (_pickedImage != null) {
        photoUrl = await _storageService.uploadProfileImage(uid, _pickedImage!);
      }

      final data = {
        'name': _nameCtrl.text.trim(),
        'bio': _bioCtrl.text.trim(),
        'specialty': _specialtyCtrl.text.trim(),
        'phone': _phoneCtrl.text.trim(),
        'year': _selectedYear,
        if (photoUrl != null) 'photoUrl': photoUrl,
      };
      await _firestoreService.updateUser(uid, data);

      final updated = authProv.user!.copyWith(
        name: _nameCtrl.text.trim(),
        bio: _bioCtrl.text.trim(),
        specialty: _specialtyCtrl.text.trim(),
        phone: _phoneCtrl.text.trim(),
        year: _selectedYear,
        photoUrl: photoUrl,
      );
      authProv.updateLocalUser(updated);

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Профиль жаңартылды'),
            backgroundColor: AppColors.success,
          ),
        );
        Navigator.pop(context);
      }
    } catch (_) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Сақтау мүмкін болмады'),
            backgroundColor: AppColors.error,
          ),
        );
      }
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final user = context.watch<AuthProvider>().user;

    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(title: const Text('Профильді өңдеу')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Form(
          key: _formKey,
          child: Column(
            children: [
              Center(
                child: GestureDetector(
                  onTap: _pickImage,
                  child: Stack(
                    children: [
                      CircleAvatar(
                        radius: 52,
                        backgroundColor: AppColors.primarySurface,
                        backgroundImage: _pickedImage != null
                            ? FileImage(_pickedImage!)
                            : (user?.photoUrl != null && user!.photoUrl!.isNotEmpty
                                    ? NetworkImage(user.photoUrl!)
                                    : null)
                                as ImageProvider?,
                        child: (_pickedImage == null &&
                                (user?.photoUrl == null ||
                                    user!.photoUrl!.isEmpty))
                            ? Text(
                                user?.name.isNotEmpty == true
                                    ? user!.name[0].toUpperCase()
                                    : '?',
                                style: GoogleFonts.nunito(
                                  fontSize: 36,
                                  fontWeight: FontWeight.w800,
                                  color: AppColors.primary,
                                ),
                              )
                            : null,
                      ),
                      Positioned(
                        right: 0,
                        bottom: 0,
                        child: Container(
                          width: 32,
                          height: 32,
                          decoration: const BoxDecoration(
                            color: AppColors.primary,
                            shape: BoxShape.circle,
                          ),
                          child: const Icon(Icons.camera_alt_rounded,
                              color: AppColors.white, size: 16),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 24),
              CustomTextField(
                label: 'Аты-жөні',
                controller: _nameCtrl,
                prefixIcon: Icons.person_outline_rounded,
                validator: (v) =>
                    v == null || v.isEmpty ? 'Аты-жөнді енгізіңіз' : null,
              ),
              const SizedBox(height: 14),
              CustomTextField(
                label: 'Мамандық',
                hint: 'Мысалы: Бухгалтерия',
                controller: _specialtyCtrl,
                prefixIcon: Icons.work_outline_rounded,
              ),
              const SizedBox(height: 14),
              CustomTextField(
                label: 'Телефон',
                hint: '+7 777 000 0000',
                controller: _phoneCtrl,
                prefixIcon: Icons.phone_outlined,
                keyboardType: TextInputType.phone,
              ),
              const SizedBox(height: 14),
              CustomTextField(
                label: 'Өзіңіз туралы',
                hint: 'Қысқаша өзіңізді таныстырыңыз...',
                controller: _bioCtrl,
                maxLines: 3,
                prefixIcon: Icons.info_outline_rounded,
              ),
              const SizedBox(height: 14),
              if (user?.role == 'student') ...[
                DropdownButtonFormField<String>(
                  value: _selectedYear,
                  decoration: const InputDecoration(
                    labelText: 'Курс',
                    prefixIcon: Icon(Icons.school_outlined, size: 20),
                  ),
                  items: ['1', '2', '3', '4'].map((y) {
                    return DropdownMenuItem(
                        value: y,
                        child: Text('$y курс', style: GoogleFonts.nunito()));
                  }).toList(),
                  onChanged: (v) => setState(() => _selectedYear = v!),
                ),
                const SizedBox(height: 14),
              ],
              const SizedBox(height: 24),
              PrimaryButton(
                label: 'Сақтау',
                isLoading: _isLoading,
                onPressed: _save,
              ),
              const SizedBox(height: 32),
            ],
          ),
        ),
      ),
    );
  }
}
