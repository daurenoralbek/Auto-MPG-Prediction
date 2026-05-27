import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../config/app_colors.dart';

class HelpCenterScreen extends StatelessWidget {
  const HelpCenterScreen({super.key});

  static const _faqs = [
    _FaqData(
      question: 'Ментор қалай табамын?',
      answer:
          '"Менторлар" бөліміне кіріп, санат бойынша іздеңіз немесе іздеу жолағын пайдаланыңыз.',
    ),
    _FaqData(
      question: 'Брондауымды қалай жіберемін?',
      answer:
          'Ментор профиліне кіріп, "Брондау" батырмасын басыңыз. Күн мен уақытты таңдаңыз.',
    ),
    _FaqData(
      question: 'Менторға хабар жіберуге болады ма?',
      answer:
          'Иә! Ментор профиліндегі "Хабар жазу" батырмасы арқылы немесе "Хабарлар" бөлімінен байланысуға болады.',
    ),
    _FaqData(
      question: 'Таңдаулылар тізімін қалай басқарамын?',
      answer:
          'Ментор картасындағы жүрек белгісін басыңыз. Таңдаулылар "Таңдаулылар" бөлімінде сақталады.',
    ),
    _FaqData(
      question: 'Профиль суретімді қалай өзгертемін?',
      answer:
          '"Профиль" → "Профильді өңдеу" бөліміне кіріп, аватарды басыңыз. Камерадан немесе галереядан сурет таңдаңыз.',
    ),
    _FaqData(
      question: 'Брондауды болдырмауға болады ма?',
      answer:
          '"Брондау тарихы" бөліміндегі күтудегі брондауды жою белгісін (қоқыс жәшігі) басыңыз.',
    ),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(title: const Text('Анықтама орталығы')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              gradient: AppColors.primaryGradient,
              borderRadius: BorderRadius.circular(16),
            ),
            child: Row(
              children: [
                const Icon(Icons.support_agent_rounded,
                    color: AppColors.white, size: 36),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Сұрақтарыңыз бар ма?',
                        style: GoogleFonts.nunito(
                          fontSize: 15,
                          fontWeight: FontWeight.w700,
                          color: AppColors.white,
                        ),
                      ),
                      Text(
                        'Жиі қойылатын сұрақтар',
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
          const SizedBox(height: 16),
          ...(_faqs.map((f) => _FaqCard(data: f))),
        ],
      ),
    );
  }
}

class _FaqData {
  final String question;
  final String answer;
  const _FaqData({required this.question, required this.answer});
}

class _FaqCard extends StatefulWidget {
  final _FaqData data;
  const _FaqCard({required this.data});

  @override
  State<_FaqCard> createState() => _FaqCardState();
}

class _FaqCardState extends State<_FaqCard> {
  bool _expanded = false;

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      decoration: BoxDecoration(
        color: AppColors.white,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppColors.divider),
      ),
      child: ExpansionTile(
        tilePadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
        childrenPadding:
            const EdgeInsets.fromLTRB(16, 0, 16, 14),
        onExpansionChanged: (v) => setState(() => _expanded = v),
        leading: Container(
          width: 32,
          height: 32,
          decoration: BoxDecoration(
            color: _expanded ? AppColors.primary : AppColors.primarySurface,
            shape: BoxShape.circle,
          ),
          child: Icon(
            Icons.help_outline_rounded,
            size: 16,
            color: _expanded ? AppColors.white : AppColors.primary,
          ),
        ),
        title: Text(
          widget.data.question,
          style: GoogleFonts.nunito(
            fontSize: 14,
            fontWeight: FontWeight.w700,
            color: AppColors.textPrimary,
          ),
        ),
        trailing: Icon(
          _expanded ? Icons.keyboard_arrow_up : Icons.keyboard_arrow_down,
          color: AppColors.textHint,
        ),
        children: [
          Text(
            widget.data.answer,
            style: GoogleFonts.nunito(
              fontSize: 13,
              color: AppColors.textSecondary,
              height: 1.6,
            ),
          ),
        ],
      ),
    );
  }
}
