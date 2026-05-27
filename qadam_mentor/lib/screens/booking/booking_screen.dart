import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:intl/intl.dart';
import '../../config/app_colors.dart';
import '../../config/app_constants.dart';
import '../../config/app_routes.dart';
import '../../models/booking_model.dart';
import '../../models/mentor_model.dart';
import '../../providers/auth_provider.dart';
import '../../providers/booking_provider.dart';
import '../../widgets/common/custom_button.dart';
import '../../widgets/common/custom_text_field.dart';

class BookingScreen extends StatefulWidget {
  const BookingScreen({super.key});

  @override
  State<BookingScreen> createState() => _BookingScreenState();
}

class _BookingScreenState extends State<BookingScreen> {
  final _topicCtrl = TextEditingController();
  final _messageCtrl = TextEditingController();
  DateTime? _selectedDate;
  String? _selectedTime;
  final _formKey = GlobalKey<FormState>();

  @override
  void dispose() {
    _topicCtrl.dispose();
    _messageCtrl.dispose();
    super.dispose();
  }

  Future<void> _pickDate() async {
    final now = DateTime.now();
    final picked = await showDatePicker(
      context: context,
      initialDate: now.add(const Duration(days: 1)),
      firstDate: now,
      lastDate: now.add(const Duration(days: 30)),
      locale: const Locale('kk'),
      builder: (ctx, child) => Theme(
        data: ThemeData.light().copyWith(
          colorScheme: const ColorScheme.light(primary: AppColors.primary),
        ),
        child: child!,
      ),
    );
    if (picked != null) setState(() => _selectedDate = picked);
  }

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) return;
    if (_selectedDate == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Күнді таңдаңыз'), backgroundColor: AppColors.error),
      );
      return;
    }
    if (_selectedTime == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Уақытты таңдаңыз'), backgroundColor: AppColors.error),
      );
      return;
    }

    final mentor =
        ModalRoute.of(context)!.settings.arguments as MentorModel;
    final user = context.read<AuthProvider>().user!;

    final booking = BookingModel(
      id: '',
      studentId: user.uid,
      mentorId: mentor.uid,
      studentName: user.name,
      mentorName: mentor.name,
      mentorPhoto: mentor.photoUrl,
      studentPhoto: user.photoUrl,
      date: _selectedDate!,
      timeSlot: _selectedTime!,
      topic: _topicCtrl.text.trim(),
      message: _messageCtrl.text.trim(),
      status: AppConstants.statusPending,
      createdAt: DateTime.now(),
    );

    final id = await context.read<BookingProvider>().createBooking(booking);
    if (!mounted) return;
    if (id != null) {
      Navigator.pushReplacementNamed(context, AppRoutes.requestSubmitted);
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Брондау жіберілмеді. Қайталап көріңіз.'),
          backgroundColor: AppColors.error,
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final mentor =
        ModalRoute.of(context)!.settings.arguments as MentorModel;
    final isLoading = context.watch<BookingProvider>().isLoading;

    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(title: const Text('Кеңес брондау')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Mentor info card
              Container(
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                  gradient: AppColors.primaryGradient,
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Row(
                  children: [
                    Container(
                      width: 50,
                      height: 50,
                      decoration: BoxDecoration(
                        color: AppColors.white.withOpacity(0.3),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Center(
                        child: Text(
                          mentor.name.isNotEmpty ? mentor.name[0] : '?',
                          style: GoogleFonts.nunito(
                            fontSize: 22,
                            fontWeight: FontWeight.w800,
                            color: AppColors.white,
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            mentor.name,
                            style: GoogleFonts.nunito(
                              fontSize: 15,
                              fontWeight: FontWeight.w700,
                              color: AppColors.white,
                            ),
                          ),
                          Text(
                            mentor.specialty,
                            style: GoogleFonts.nunito(
                              fontSize: 12,
                              color: AppColors.white.withOpacity(0.8),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 20),

              _SectionLabel('Тақырып'),
              const SizedBox(height: 8),
              CustomTextField(
                label: 'Кеңес тақырыбы',
                hint: 'Мысалы: IELTS Speaking дайындығы',
                controller: _topicCtrl,
                prefixIcon: Icons.topic_outlined,
                validator: (v) => v == null || v.isEmpty ? 'Тақырыпты енгізіңіз' : null,
              ),
              const SizedBox(height: 16),

              _SectionLabel('Күнді таңдаңыз'),
              const SizedBox(height: 8),
              GestureDetector(
                onTap: _pickDate,
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
                  decoration: BoxDecoration(
                    color: AppColors.white,
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: AppColors.divider),
                  ),
                  child: Row(
                    children: [
                      const Icon(Icons.calendar_today_outlined,
                          color: AppColors.textSecondary, size: 20),
                      const SizedBox(width: 12),
                      Text(
                        _selectedDate == null
                            ? 'Күн таңдаңыз'
                            : DateFormat('dd MMMM yyyy', 'kk').format(_selectedDate!),
                        style: GoogleFonts.nunito(
                          fontSize: 14,
                          color: _selectedDate == null
                              ? AppColors.textHint
                              : AppColors.textPrimary,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 16),

              _SectionLabel('Уақытты таңдаңыз'),
              const SizedBox(height: 8),
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: AppConstants.timeSlots.map((t) {
                  final selected = _selectedTime == t;
                  return GestureDetector(
                    onTap: () => setState(() => _selectedTime = t),
                    child: AnimatedContainer(
                      duration: const Duration(milliseconds: 200),
                      padding: const EdgeInsets.symmetric(
                          horizontal: 16, vertical: 10),
                      decoration: BoxDecoration(
                        color: selected ? AppColors.primary : AppColors.white,
                        borderRadius: BorderRadius.circular(10),
                        border: Border.all(
                          color: selected ? AppColors.primary : AppColors.divider,
                        ),
                      ),
                      child: Text(
                        t,
                        style: GoogleFonts.nunito(
                          fontSize: 13,
                          fontWeight: FontWeight.w700,
                          color: selected ? AppColors.white : AppColors.textPrimary,
                        ),
                      ),
                    ),
                  );
                }).toList(),
              ),
              const SizedBox(height: 16),

              _SectionLabel('Қосымша хабар (міндетті емес)'),
              const SizedBox(height: 8),
              CustomTextField(
                label: 'Хабар',
                hint: 'Менторға қосымша ақпарат жазыңыз...',
                controller: _messageCtrl,
                maxLines: 3,
              ),
              const SizedBox(height: 28),

              PrimaryButton(
                label: 'Сұрау жіберу',
                isLoading: isLoading,
                onPressed: _submit,
              ),
              const SizedBox(height: 24),
            ],
          ),
        ),
      ),
    );
  }
}

class _SectionLabel extends StatelessWidget {
  final String text;
  const _SectionLabel(this.text);

  @override
  Widget build(BuildContext context) {
    return Text(
      text,
      style: GoogleFonts.nunito(
        fontSize: 14,
        fontWeight: FontWeight.w700,
        color: AppColors.textPrimary,
      ),
    );
  }
}
