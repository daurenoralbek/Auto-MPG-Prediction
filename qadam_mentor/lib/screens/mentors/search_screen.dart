import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../config/app_colors.dart';
import '../../config/app_routes.dart';
import '../../providers/mentor_provider.dart';
import '../../widgets/common/empty_state_widget.dart';
import '../../widgets/mentor_card.dart';

class SearchScreen extends StatefulWidget {
  const SearchScreen({super.key});

  @override
  State<SearchScreen> createState() => _SearchScreenState();
}

class _SearchScreenState extends State<SearchScreen> {
  final _ctrl = TextEditingController();
  bool _hasSearched = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance
        .addPostFrameCallback((_) => _focusNode.requestFocus());
  }

  final _focusNode = FocusNode();

  @override
  void dispose() {
    _ctrl.dispose();
    _focusNode.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final mentors = context.watch<MentorProvider>();
    final results = _hasSearched ? mentors.mentors : [];

    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        title: TextField(
          controller: _ctrl,
          focusNode: _focusNode,
          autofocus: true,
          decoration: const InputDecoration(
            hintText: 'Ментор іздеу...',
            border: InputBorder.none,
            filled: false,
          ),
          onChanged: (v) {
            context.read<MentorProvider>().search(v);
            setState(() => _hasSearched = v.isNotEmpty);
          },
        ),
        actions: [
          if (_ctrl.text.isNotEmpty)
            IconButton(
              icon: const Icon(Icons.close),
              onPressed: () {
                _ctrl.clear();
                context.read<MentorProvider>().search('');
                setState(() => _hasSearched = false);
              },
            ),
        ],
      ),
      body: !_hasSearched
          ? Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(Icons.search_rounded,
                      size: 64, color: AppColors.divider),
                  const SizedBox(height: 12),
                  Text(
                    'Ментор аты немесе мамандық енгізіңіз',
                    style: GoogleFonts.nunito(
                      fontSize: 14,
                      color: AppColors.textHint,
                    ),
                  ),
                ],
              ),
            )
          : results.isEmpty
              ? const EmptyStateWidget(
                  icon: Icons.search_off_rounded,
                  title: 'Нәтиже табылмады',
                  subtitle: 'Басқа сөздерді қолданып көріңіз',
                )
              : ListView.builder(
                  padding: const EdgeInsets.all(16),
                  itemCount: results.length,
                  itemBuilder: (ctx, i) {
                    final m = mentors.mentors[i];
                    return MentorCard(
                      mentor: m,
                      onTap: () => Navigator.pushNamed(
                        ctx,
                        AppRoutes.mentorDetail,
                        arguments: m,
                      ),
                    );
                  },
                ),
    );
  }
}
